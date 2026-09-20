"""Tests for passive tok/s stats and session detection.

The acceptance criteria require driving the app against the fake llama
server over real HTTP and asserting on stored sessions.
"""

import asyncio

import httpx
import pytest

from monitor.telemetry.real import RealLLaMAStatsSource
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
        # First sample: cumulative only, no rates yet
        assert "generated_tokens_rate" not in first
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 120, "generated_tokens_total": 45}},
            )
        
        second = await source.collect()
        assert second["prompt_tokens"] == 120
        assert second["generated_tokens"] == 45
        assert second["prompt_tokens_rate"] == 120
        assert second["generated_tokens_rate"] == 45


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
        
        await source.collect()
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"speculative_accepts_total": 23}},
            )
        second = await source.collect()
        
        assert second["speculative_accepts"] == 100
        assert second["speculative_accepts_rate"] == 23


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
async def test_no_session_without_activity(test_db):
    """No tokens moving -> no session rows."""
    await init_db()
    tracker = SessionTracker(model_name="test-model", idle_timeout=0.1)
    
    await tracker.observe({"generated_tokens_rate": 0, "prompt_tokens_rate": 0})
    await tracker.observe(None)
    
    assert await get_active_session() is None
    assert await get_sessions_history() == []
