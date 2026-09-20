"""Tests for fake LLaMA server."""

import pytest
import httpx
from tests.fake_llama_server import FakeLLaMAServer


@pytest.mark.asyncio
async def test_fake_llama_server_metrics():
    """Test fake LLaMA server /metrics endpoint."""
    server = FakeLLaMAServer(port=18081)
    await server.start()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server.url}/metrics")
            
            assert response.status_code == 200
            assert "llamacpp:prompt_tokens_total" in response.text
            assert "llamacpp:tokens_generated_total" in response.text
            assert "llamacpp:speculative_accepts_total" in response.text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_fake_llama_server_increment():
    """Test fake LLaMA server increment endpoint."""
    server = FakeLLaMAServer(port=18082)
    await server.start()
    
    try:
        async with httpx.AsyncClient() as client:
            # Initial metrics
            response = await client.get(f"{server.url}/metrics")
            assert "llamacpp:prompt_tokens_total 0" in response.text
            
            # Increment
            response = await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 100}}
            )
            assert response.status_code == 200
            
            # Check incremented value
            response = await client.get(f"{server.url}/metrics")
            assert "llamacpp:prompt_tokens_total 100" in response.text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_fake_llama_server_reset():
    """Test fake LLaMA server reset endpoint."""
    server = FakeLLaMAServer(port=18083)
    await server.start()
    
    try:
        async with httpx.AsyncClient() as client:
            # Increment first
            await client.post(
                f"{server.url}/metrics/inc",
                json={"increments": {"prompt_tokens_total": 500}}
            )
            
            # Reset
            response = await client.post(
                f"{server.url}/metrics/reset",
                json={"counters": {"prompt_tokens_total": 0}}
            )
            assert response.status_code == 200
            
            # Check reset value
            response = await client.get(f"{server.url}/metrics")
            assert "llamacpp:prompt_tokens_total 0" in response.text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_fake_llama_server_context_manager():
    """Test fake LLaMA server as async context manager."""
    async with FakeLLaMAServer(port=18084) as server:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server.url}/metrics")
            assert response.status_code == 200
