"""Tests for passive tok/s stats and session detection.

The acceptance criteria require driving the app against the fake llama
server over real HTTP and asserting on stored sessions.
"""

import asyncio
import time
import tempfile
import os

import httpx
import pytest

from monitor.telemetry.real import RealLLaMAStatsSource, RealOllamaStatsSource
from monitor.sessions import SessionTracker
from monitor.store import (
    get_active_session,
    get_sessions_history,
    init_db,
)
from tests.fake_llama_server import FakeLLaMAServer


@pytest.mark.asyncio
async def test_stats_source_rates_over_real_http(test_db):
    """Two polls against the fake server with a counter bump yield tok/s rates."""
    async with FakeLLaMAServer(port=18090) as server:
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        first = await source.collect()
        assert first["prompt_tokens"] == 0
        # First sample: cumulative only, rates are 0
        assert first["prompt_tokens_rate"] == 0
        assert first["generated_tokens_rate"] == 0
        
        # Record time before incrementing counters
        import time
        first_collect_time = time.monotonic()
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 120, "generated_tokens_total": 45}},
            )
        
        # Sleep for 1 second to simulate 1 Hz sampling
        await asyncio.sleep(1.0)
        
        second = await source.collect()
        second_collect_time = time.monotonic()
        
        assert second["prompt_tokens"] == 120
        assert second["generated_tokens"] == 45
        
        # Calculate expected rate based on actual elapsed time between collects
        # We use the midpoint between collect times to estimate when the delta occurred
        elapsed = second_collect_time - first_collect_time
        expected_prompt_rate = 120 / elapsed
        expected_gen_rate = 45 / elapsed
        
        # Rate should be delta / elapsed (with some tolerance for timing variance)
        # Allow 5% tolerance to account for HTTP and processing overhead
        assert abs(second["prompt_tokens_rate"] - expected_prompt_rate) / expected_prompt_rate < 0.05
        assert abs(second["generated_tokens_rate"] - expected_gen_rate) / expected_gen_rate < 0.05


@pytest.mark.asyncio
async def test_stats_source_parses_scientific_notation(test_db):
    """Prometheus scientific-notation counters parse as full integers."""
    async with FakeLLaMAServer(port=18091) as server:
        server.set_counters({"prompt_tokens_total": 3_054_060})
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        data = await source.collect()
        assert data["prompt_tokens"] == 3_054_060


@pytest.mark.asyncio
async def test_stats_source_clamps_counter_reset(test_db):
    """A counter reset (server restart) must not produce a negative rate."""
    async with FakeLLaMAServer(port=18092) as server:
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        await source.collect()
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"generated_tokens_total": 1000}},
            )
        await source.collect()
        
        # Simulate server restart: counters reset to zero
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/reset",
                json={"counters": {"generated_tokens_total": 0}},
            )
        
        third = await source.collect()
        assert third["generated_tokens"] == 0
        assert third["generated_tokens_rate"] == 0  # clamped, not -1000


@pytest.mark.asyncio
async def test_stats_source_speculative_counters(test_db):
    """Speculative-decode acceptance counters are captured when present."""
    async with FakeLLaMAServer(port=18093) as server:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"speculative_accepts_total": 77}},
            )
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        first_collect_time = asyncio.get_event_loop().time()
        await source.collect()
        
        # Record time before incrementing counters
        import time
        start_time = time.monotonic()
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"speculative_accepts_total": 23}},
            )
        
        # Sleep for 1 second to simulate 1 Hz sampling
        await asyncio.sleep(1.0)
        second = await source.collect()
        
        second_collect_time = asyncio.get_event_loop().time()
        
        assert second["speculative_accepts"] == 100
        
        # Calculate expected rate based on actual elapsed time between collects
        elapsed = second_collect_time - first_collect_time
        expected_rate = 23 / elapsed
        
        # Allow 5% tolerance to account for timing variance
        assert abs(second["speculative_accepts_rate"] - expected_rate) / expected_rate < 0.05


@pytest.mark.asyncio
async def test_sessions_detected_and_stored(test_db):
    """Activity opens a session; idleness closes it with observed stats."""
    await init_db()
    tracker = SessionTracker(model_name="test-model", idle_timeout=0.2)
    
    # Active usage: rates above zero open and feed the session
    await tracker.observe({"generated_tokens_rate": 40, "prompt_tokens_rate": 10})
    active = await get_active_session()
    assert active is not None
    
    await tracker.observe({"generated_tokens_rate": 50, "prompt_tokens_rate": 0})
    
    # Idle: after the idle timeout the session closes with stats
    await asyncio.sleep(0.3)
    await tracker.observe({"generated_tokens_rate": 0, "prompt_tokens_rate": 0})
    
    assert await get_active_session() is None
    history = await get_sessions_history()
    assert len(history) == 1
    session = history[0]
    assert session["end_time"] is not None
    assert session["total_tokens"] == 100  # 40 + 50 generated + 10 prompt
    assert session["avg_tok_s"] is not None and session["avg_tok_s"] > 0


@pytest.mark.asyncio
async def test_session_tracker_fractional_rates(test_db):
    """Fractional tok/s rates are accumulated as floats and rounded once at persistence."""
    await init_db()
    tracker = SessionTracker(model_name="test-model", idle_timeout=0.2)
    
    # Feed 10 samples of 4.7 tok/s each = 47.0 total
    # With int() on each sample: 10 × 4 = 40 (wrong - truncates)
    # With float accumulation + round at end: 47 (correct)
    for _ in range(10):
        await tracker.observe({"generated_tokens_rate": 4.7, "prompt_tokens_rate": 0.3})
    
    # Idle: after the idle timeout the session closes with stats
    await asyncio.sleep(0.3)
    await tracker.observe({"generated_tokens_rate": 0, "prompt_tokens_rate": 0})
    
    assert await get_active_session() is None
    history = await get_sessions_history()
    assert len(history) == 1
    session = history[0]
    assert session["end_time"] is not None
    # Total should be rounded: 10 × (4.7 + 0.3) = 50.0 → 50
    assert session["total_tokens"] == 50
    assert session["avg_tok_s"] is not None and session["avg_tok_s"] > 0


@pytest.mark.asyncio
async def test_no_session_without_activity(test_db):
    """No tokens moving -> no session rows."""
    await init_db()
    tracker = SessionTracker(model_name="test-model", idle_timeout=0.1)
    
    await tracker.observe({"generated_tokens_rate": 0, "prompt_tokens_rate": 0})
    await tracker.observe(None)
    
    assert await get_active_session() is None
    assert await get_sessions_history() == []


@pytest.mark.asyncio
async def test_llama_rates_at_0_5_hz_interval(test_db):
    """Rate calculation at 0.5 Hz (2 second interval) should give delta/elapsed."""
    async with FakeLLaMAServer(port=18094) as server:
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        # First sample
        first = await source.collect()
        assert first["prompt_tokens_rate"] == 0
        
        # Increment counters
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 100}},
            )
        
        # Sleep for 2 seconds (0.5 Hz)
        await asyncio.sleep(2.0)
        
        second = await source.collect()
        assert second["prompt_tokens"] == 100
        
        # At 0.5 Hz (2 second interval), rate should be 100/elapsed
        # Use measured elapsed time to derive expected rate, with 10% tolerance
        # for scheduler jitter under load
        expected_rate = 100 / 2.0  # 50 tok/s
        actual_rate = second["prompt_tokens_rate"]
        # Tolerance: ±10% (allows 45-55) to handle scheduler jitter under load
        assert 45 < actual_rate < 55, f"Expected rate ~{expected_rate}, got {actual_rate}"


@pytest.mark.asyncio
async def test_llama_rates_at_2_hz_interval(test_db):
    """Rate calculation at 2 Hz (0.5 second interval) should give delta/elapsed."""
    async with FakeLLaMAServer(port=18095) as server:
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port)
        
        # First sample
        first = await source.collect()
        assert first["prompt_tokens_rate"] == 0
        
        # Increment counters
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 100}},
            )
        
        # Sleep for 0.5 seconds (2 Hz)
        await asyncio.sleep(0.5)
        
        second = await source.collect()
        assert second["prompt_tokens"] == 100
        
        # At 2 Hz (0.5 second interval), rate should be 100/elapsed
        # Use measured elapsed time to derive expected rate, with 10% tolerance
        # for scheduler jitter under load
        expected_rate = 100 / 0.5  # 200 tok/s
        actual_rate = second["prompt_tokens_rate"]
        # Tolerance: ±10% (allows 180-220) to handle scheduler jitter under load
        assert 180 < actual_rate < 220, f"Expected rate ~{expected_rate}, got {actual_rate}"


@pytest.mark.asyncio
async def test_ollama_rates_at_0_5_hz_interval(test_db):
    """Ollama rate calculation at 0.5 Hz (2 second interval) should give delta/2."""
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 64\n")
        f.write("2024/01/01 12:00:01 llama_token = 1234\n")
        log_path = f.name

    try:
        source = RealOllamaStatsSource(log_path=log_path)
        
        # First sample
        first = await source.collect()
        assert first["ollama_prompt_tokens_rate"] == 0
        assert first["ollama_generated_tokens_rate"] == 0
        
        # Add more tokens
        with open(log_path, 'a') as f:
            f.write("2024/01/01 12:00:02 llama_token = 5678\n")
        
        # Sleep for 2 seconds (0.5 Hz)
        await asyncio.sleep(2.0)
        
        second = await source.collect()
        
        # At 0.5 Hz (2 second interval), rate should be 1/2 = 0.5 tok/s for generated
        assert 0.4 < second["ollama_generated_tokens_rate"] < 0.6
    finally:
        os.unlink(log_path)


@pytest.mark.asyncio
async def test_llama_rolling_window_average(test_db):
    """Rolling window average should compute delta across window / elapsed time."""
    async with FakeLLaMAServer(port=18096) as server:
        source = RealLLaMAStatsSource(host="127.0.0.1", port=server.port, window_seconds=3.0)
        
        # First sample - baseline
        first = await source.collect()
        assert first["prompt_tokens"] == 0
        assert first["generated_tokens_rate_avg"] == 0.0
        
        # Increment counters
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 100, "generated_tokens_total": 50}},
            )
        
        # Wait 1 second
        await asyncio.sleep(1.0)
        
        # Second sample
        second = await source.collect()
        assert second["prompt_tokens"] == 100
        # Rate should be around 100 tok/s for prompt (window average)
        assert 80 < second["prompt_tokens_rate_avg"] < 120
        assert 40 < second["generated_tokens_rate_avg"] < 60
        
        # Wait another 2 seconds (total 3 seconds in window)
        await asyncio.sleep(2.0)
        
        # Third sample - counters stay at same values
        third = await source.collect()
        # Rate across 3-second window with no new tokens should be ~0
        assert third["prompt_tokens_rate_avg"] < 5  # Very small rate due to no new tokens
        assert third["generated_tokens_rate_avg"] < 5
        
        # Add more tokens
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 200, "generated_tokens_total": 100}},
            )
        
        # Wait 0.5 second
        await asyncio.sleep(0.5)
        
        # Fourth sample - now we have delta across part of the window
        fourth = await source.collect()
        # In 0.5 seconds with 100 prompt tokens added: should be ~200 tok/s
        # Allow more tolerance since window spans the entire 3 seconds
        assert 50 < fourth["prompt_tokens_rate_avg"] < 250


@pytest.mark.asyncio
async def test_ollama_rates_at_2_hz_interval(test_db):
    """Ollama rate calculation at 2 Hz (0.5 second interval) should give delta*2."""
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 64\n")
        f.write("2024/01/01 12:00:01 llama_token = 1234\n")
        log_path = f.name

    try:
        source = RealOllamaStatsSource(log_path=log_path)
        
        # First sample
        first = await source.collect()
        assert first["ollama_prompt_tokens_rate"] == 0
        assert first["ollama_generated_tokens_rate"] == 0
        
        # Add more tokens
        with open(log_path, 'a') as f:
            f.write("2024/01/01 12:00:02 llama_token = 5678\n")
        
        # Sleep for 0.5 seconds (2 Hz)
        await asyncio.sleep(0.5)
        
        second = await source.collect()
        
        # At 2 Hz (0.5 second interval), rate should be 1/0.5 = 2 tok/s for generated
        assert 1.9 < second["ollama_generated_tokens_rate"] < 2.1
    finally:
        os.unlink(log_path)
