"""Tests for RealOllamaStatsSource."""

import asyncio
import pytest
import tempfile
import os
from monitor.telemetry.real import RealOllamaStatsSource


class TestRealOllamaStatsSource:
    """Tests for RealOllamaStatsSource with real log parsing."""

    @pytest.mark.asyncio
    async def test_collect_returns_namespaced_keys(self):
        """Test that RealOllamaStatsSource returns namespaced keys."""
        # Create a temporary log file with sample ollama log entries
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 64\n")
            f.write("2024/01/01 12:00:01 llama_token = 1234\n")
            f.write("2024/01/01 12:00:02 llama_token = 5678\n")
            f.write("2024/01/01 12:00:03 llama_token = 9012\n")
            log_path = f.name

        try:
            source = RealOllamaStatsSource(log_path=log_path)
            data = await source.collect()

            # Check namespaced keys are present
            assert "ollama_prompt_tokens" in data
            assert "ollama_generated_tokens" in data
            assert "ollama_prompt_tokens_rate" in data
            assert "ollama_generated_tokens_rate" in data

            # Check old keys are NOT present
            assert "prompt_tokens" not in data
            assert "generated_tokens" not in data
            assert "prompt_tokens_rate" not in data
            assert "generated_tokens_rate" not in data

            # Check values
            assert data["ollama_prompt_tokens"] == 64
            assert data["ollama_generated_tokens"] == 3  # 3 llama_token entries
        finally:
            os.unlink(log_path)

    @pytest.mark.asyncio
    async def test_collect_empty_when_no_log_file(self):
        """Test that RealOllamaStatsSource returns empty stats when log file doesn't exist."""
        source = RealOllamaStatsSource(log_path="/nonexistent/path/ollama.log")
        data = await source.collect()

        assert data["ollama_prompt_tokens"] == 0
        assert data["ollama_generated_tokens"] == 0
        assert data["ollama_prompt_tokens_rate"] == 0
        assert data["ollama_generated_tokens_rate"] == 0

    @pytest.mark.asyncio
    async def test_collect_rate_calculation(self):
        """Test that rates are calculated correctly between samples."""
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 64\n")
            f.write("2024/01/01 12:00:01 llama_token = 1234\n")
            f.write("2024/01/01 12:00:02 llama_token = 5678\n")
            log_path = f.name

        try:
            source = RealOllamaStatsSource(log_path=log_path)

            # First call - should get absolute values, rate should be 0 for first sample
            data1 = await source.collect()
            assert data1["ollama_prompt_tokens"] == 64
            assert data1["ollama_generated_tokens"] == 2
            assert data1["ollama_prompt_tokens_rate"] == 0  # First sample, no rate yet
            assert data1["ollama_generated_tokens_rate"] == 0  # First sample, no rate yet

            # Add more tokens to log file
            with open(log_path, 'a') as f:
                f.write("2024/01/01 12:00:03 llama_token = 9012\n")
                f.write("2024/01/01 12:00:04 llama_token = 2345\n")

            # Sleep for 1 second to simulate 1 Hz sampling
            await asyncio.sleep(1.0)

            # Second call - should get rate of new tokens (2 tokens over 1 second = 2 tok/s)
            data2 = await source.collect()
            assert data2["ollama_prompt_tokens"] == 64
            assert data2["ollama_generated_tokens"] == 4  # 2 more tokens
            assert data2["ollama_prompt_tokens_rate"] == 0  # No change in prompt tokens
            assert abs(data2["ollama_generated_tokens_rate"] - 2) < 0.1  # 2 tokens / 1 second
        finally:
            os.unlink(log_path)

    @pytest.mark.asyncio
    async def test_empty_log_file(self):
        """Test with an empty log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            source = RealOllamaStatsSource(log_path=log_path)
            data = await source.collect()

            assert data["ollama_prompt_tokens"] == 0
            assert data["ollama_generated_tokens"] == 0
        finally:
            os.unlink(log_path)

    @pytest.mark.asyncio
    async def test_log_file_with_only_prompt_tokens(self):
        """Test log file with only n_tokens entries (no generated tokens)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 128\n")
            f.write("2024/01/01 12:00:01 llama_new_context: n_tokens = 256\n")  # Should use last
            log_path = f.name

        try:
            source = RealOllamaStatsSource(log_path=log_path)
            data = await source.collect()

            assert data["ollama_prompt_tokens"] == 256  # Last context
            assert data["ollama_generated_tokens"] == 0
        finally:
            os.unlink(log_path)

    @pytest.mark.asyncio
    async def test_ollama_rolling_window_average(self):
        """Ollama rolling window average should compute delta across window / elapsed time."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("2024/01/01 12:00:00 llama_new_context: n_tokens = 64\n")
            f.write("2024/01/01 12:00:01 llama_token = 1234\n")
            log_path = f.name

        try:
            source = RealOllamaStatsSource(log_path=log_path, window_seconds=3.0)
            
            # First sample
            first = await source.collect()
            assert first["ollama_prompt_tokens_rate_avg"] == 0.0
            assert first["ollama_generated_tokens_rate_avg"] == 0.0
            
            # Add more tokens
            with open(log_path, 'a') as f:
                f.write("2024/01/01 12:00:02 llama_token = 5678\n")
            
            # Wait 1 second
            await asyncio.sleep(1.0)
            
            # Second sample - 1 generated token in 1 second = 1 tok/s
            second = await source.collect()
            assert 0.5 < second["ollama_generated_tokens_rate_avg"] < 1.5
            
            # Wait another 2 seconds (total 3 seconds in window)
            await asyncio.sleep(2.0)
            
            # Third sample - no new tokens, rate across window should be ~0
            third = await source.collect()
            assert third["ollama_generated_tokens_rate_avg"] < 0.5  # Very small rate
            
            # Add more tokens
            with open(log_path, 'a') as f:
                f.write("2024/01/01 12:00:03 llama_token = 9012\n")
            
            # Wait 0.5 second
            await asyncio.sleep(0.5)
            
            # Fourth sample - 1 token in 0.5 seconds = 2 tok/s
            # Allow more tolerance since window spans the entire 3 seconds
            fourth = await source.collect()
            assert 0.1 < fourth["ollama_generated_tokens_rate_avg"] < 1.5
        finally:
            os.unlink(log_path)
