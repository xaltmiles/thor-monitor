"""SSE streaming helpers."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Tuple, List
import httpx


def count_tokens_in_sse_chunk(chunk: Dict[str, Any]) -> int:
    """Count tokens in an SSE chunk from streaming response.
    
    First checks usage field (authoritative), then delta.content.
    """
    # Check usage field first (authoritative for completion tokens)
    try:
        usage = chunk.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            return completion_tokens
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        if prompt_tokens > 0:
            return prompt_tokens
    except (AttributeError, TypeError):
        pass
    
    # Fall back to delta.content
    try:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            content = delta["content"]
            if content and content != "[DONE]":
                # Rough token count: 1 token ≈ 4 chars
                return len(content) // 4
    except (IndexError, KeyError):
        pass
    
    return 0


async def stream_workload(
    client: httpx.AsyncClient,
    port: int,
    prompt_text: str,
    max_tokens: int,
    stream_config: Optional[Dict[str, float]] = None
) -> Tuple[Optional[float], int, float, float]:
    """Stream a workload and return (ttft, tokens, time_before_first, elapsed_time).
    
    Args:
        client: HTTP client
        port: Server port
        prompt_text: Prompt text
        max_tokens: Maximum tokens to generate
        stream_config: Optional config for streaming behavior
        
    Returns:
        (ttft, tokens_generated, time_before_first_token, total_elapsed)
    """
    if stream_config is None:
        stream_config = {}
    
    stream_delay = stream_config.get("delay_ms", 50) / 1000.0
    first_chunk_delay = stream_config.get("first_chunk_delay_ms", 200) / 1000.0
    
    request = {
        "model": "benchmark-model",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    ttft = None
    tokens_generated = 0
    time_before_first_token = None
    request_start_time = time.time()
    
    try:
        async with client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                
                if line.strip() == "data: [DONE]":
                    break
                
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                
                chunk_time = time.time()
                
                # Track time to first token
                if ttft is None and chunk.get("choices"):
                    ttft = chunk_time - request_start_time
                    time_before_first_token = ttft
                    
                    # Wait for first chunk delay to simulate realistic TTFT
                    if first_chunk_delay > 0:
                        await asyncio.sleep(first_chunk_delay)
                
                # Count tokens
                chunk_tokens = count_tokens_in_sse_chunk(chunk)
                if chunk_tokens > 0:
                    tokens_generated += chunk_tokens
                
                # Simulate token pacing if we're after TTFT
                if ttft is not None and tokens_generated > 0:
                    if chunk_tokens > 0:
                        await asyncio.sleep(stream_delay)
        
        elapsed = time.time() - request_start_time
        return ttft, tokens_generated, time_before_first_token or elapsed, elapsed
        
    except httpx.RequestError:
        elapsed = time.time() - request_start_time
        return None, 0, elapsed, elapsed


async def run_concurrent_requests(
    port: int,
    num_requests: int,
    prompt_text: str,
    max_tokens: int
) -> Tuple[float, int, Optional[float]]:
    """Run multiple concurrent requests and return (elapsed, tokens, ttft).
    
    Returns wall-clock time, total tokens generated, and earliest TTFT.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            _run_single_request(client, port, prompt_text, max_tokens)
            for _ in range(num_requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, tuple) and r[0] is not None]
        
        if not valid_results:
            return 0.0, 0, None
        
        elapsed = max(r[0] for r in valid_results)  # Wall-clock time
        total_tokens = sum(r[1] for r in valid_results)
        ttft = min(r[2] for r in valid_results)  # Earliest TTFT
        
        return elapsed, total_tokens, ttft


async def _run_single_request(
    client: httpx.AsyncClient,
    port: int,
    prompt_text: str,
    max_tokens: int
) -> Tuple[float, int, Optional[float]]:
    """Run a single request, return (elapsed, tokens, ttft)."""
    request = {
        "model": "benchmark-model",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    ttft = None
    tokens_generated = 0
    request_start_time = time.time()
    
    try:
        async with client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                
                if line.strip() == "data: [DONE]":
                    break
                
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                
                chunk_time = time.time()
                
                if ttft is None and chunk.get("choices"):
                    ttft = chunk_time - request_start_time
                
                tokens_generated += count_tokens_in_sse_chunk(chunk)
                
    except httpx.RequestError:
        pass
    
    elapsed = time.time() - request_start_time
    return elapsed, tokens_generated, ttft
