"""Workload runners for benchmark suite."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import httpx


class WorkloadResult:
    """Result from a single workload execution."""
    
    def __init__(
        self,
        workload_name: str,
        total_time: float,
        tokens_per_second: float,
        ttft: Optional[float] = None,
        prompt_tokens: int = 0,
        generated_tokens: int = 0
    ):
        self.workload_name = workload_name
        self.total_time = total_time
        self.tokens_per_second = tokens_per_second
        self.ttft = ttft
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "workload_name": self.workload_name,
            "total_time": self.total_time,
            "tokens_per_second": self.tokens_per_second,
            "ttft": self.ttft,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }


class LongContextWorkloadRunner:
    """Long-context workload: 16k token prompt, short generation.
    
    Measures prompt-processing tok/s (tokens per second to process the prompt).
    """
    
    DEFAULT_PROMPT_TOKENS = 16384  # 16k tokens
    DEFAULT_GENERATION_TOKENS = 32
    
    def __init__(
        self,
        prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
        generation_tokens: int = DEFAULT_GENERATION_TOKENS,
        server_port: Optional[int] = None
    ):
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.server_port = server_port
    
    async def run(self, server_port: int) -> WorkloadResult:
        """Run the long-context workload against the server.
        
        Args:
            server_port: Port of the LLM server
            
        Returns:
            WorkloadResult with prompt-processing tok/s
        """
        self.server_port = server_port
        
        # Create a prompt with approximately prompt_tokens tokens
        # Rough estimate: 1 token ≈ 4 characters
        prompt_char_count = self.prompt_tokens * 4
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_char_count // 64)
        prompt_text = prompt_text[:prompt_char_count]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            request = {
                "model": "benchmark-model",
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": self.generation_tokens,
                "stream": True
            }
            
            ttft = None
            prompt_tokens_processed = 0
            gen_tokens = 0
            first_token_time = None
            
            request_start_time = time.time()
            
            try:
                async with client.stream("POST", f"http://localhost:{server_port}/v1/chat/completions", json=request) as response:
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
                        
                        # Track TTFT
                        if ttft is None and chunk.get("choices"):
                            ttft = time.time() - request_start_time
                            first_token_time = time.time()
                        
                        # Count tokens in chunk
                        chunk_tokens = self._count_tokens_in_chunk(chunk)
                        if chunk_tokens > 0:
                            if ttft is not None:
                                gen_tokens += chunk_tokens
                            else:
                                # Before first token, we're processing the prompt
                                prompt_tokens_processed += chunk_tokens
                    
                    # Calculate prompt tok/s
                    total_time = time.time() - request_start_time
                    prompt_tok_s = prompt_tokens_processed / total_time if total_time > 0 else 0
                    
                    return WorkloadResult(
                        workload_name="long-context",
                        total_time=total_time,
                        tokens_per_second=prompt_tok_s,
                        ttft=ttft,
                        prompt_tokens=prompt_tokens_processed,
                        generated_tokens=gen_tokens
                    )
                    
            except httpx.RequestError as e:
                return WorkloadResult(
                    workload_name="long-context",
                    total_time=time.time() - request_start_time,
                    tokens_per_second=0,
                    ttft=None,
                    prompt_tokens=prompt_tokens_processed,
                    generated_tokens=gen_tokens
                )
    
    def _count_tokens_in_chunk(self, chunk: Dict[str, Any]) -> int:
        """Count tokens in a chunk from the streaming response."""
        try:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                if content and content != "[DONE]":
                    # Rough token count: 1 token ≈ 4 chars
                    return len(content) // 4
        except (IndexError, KeyError):
            pass
        
        # Check usage field
        usage = chunk.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            return completion_tokens
        
        return 0


class BurstWorkloadRunner:
    """Burst workload: 4 concurrent requests, measures aggregate generation throughput."""
    
    DEFAULT_CONCURRENCY = 4
    DEFAULT_TOKENS_PER_REQUEST = 128
    
    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        tokens_per_request: int = DEFAULT_TOKENS_PER_REQUEST,
        server_port: Optional[int] = None
    ):
        self.concurrency = concurrency
        self.tokens_per_request = tokens_per_request
        self.server_port = server_port
    
    async def run(self, server_port: int) -> WorkloadResult:
        """Run the burst workload with concurrent requests.
        
        Args:
            server_port: Port of the LLM server
            
        Returns:
            WorkloadResult with aggregate generation throughput
        """
        self.server_port = server_port
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create concurrent requests
            tasks = [
                self._run_single_request(client, server_port)
                for _ in range(self.concurrency)
            ]
            
            # Run all requests concurrently
            results: List[Tuple[float, float]] = []  # (time, tokens)
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, tuple):
                    results.append(result)
            
            if not results:
                return WorkloadResult(
                    workload_name="burst",
                    total_time=0,
                    tokens_per_second=0
                )
            
            if not results:
                return WorkloadResult(
                    workload_name="burst",
                    total_time=0,
                    tokens_per_second=0
                )
            
            # Calculate aggregate throughput
            times = [t for t, _ in results if t is not None]
            total_time = max(times) if times else 0
            total_tokens = sum(tokens for _, tokens in results)
            
            # TTFT is the time to first token across all requests
            ttft_values = [ttft for ttft, _ in results if ttft is not None]
            ttft = min(ttft_values) if ttft_values else None
            
            aggregate_tok_s = total_tokens / total_time if total_time > 0 else 0
            
            return WorkloadResult(
                workload_name="burst",
                total_time=total_time,
                tokens_per_second=aggregate_tok_s,
                ttft=ttft,
                generated_tokens=total_tokens
            )
    
    async def _run_single_request(
        self,
        client: httpx.AsyncClient,
        port: int
    ) -> Tuple[Optional[float], int]:
        """Run a single request and return (ttft, tokens)."""
        request = {
            "model": "benchmark-model",
            "messages": [
                {"role": "user", "content": "Write a short story about AI."}
            ],
            "max_tokens": self.tokens_per_request,
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
                    
                    # Track TTFT
                    if ttft is None and chunk.get("choices"):
                        ttft = time.time() - request_start_time
                    
                    # Count tokens
                    chunk_tokens = self._count_tokens_in_chunk(chunk)
                    tokens_generated += chunk_tokens
                    
        except httpx.RequestError:
            pass
        
        return ttft, tokens_generated
    
    def _count_tokens_in_chunk(self, chunk: Dict[str, Any]) -> int:
        """Count tokens in a chunk from the streaming response."""
        try:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                if content and content != "[DONE]":
                    return len(content) // 4
        except (IndexError, KeyError):
            pass
        
        usage = chunk.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            return completion_tokens
        
        return 0
