"""Workload runners for benchmark suite."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import httpx

from .sse_helper import count_tokens_in_sse_chunk


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
        server_port: Optional[int] = None  # Kept for backward compatibility
    ):
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
    
    async def run(self, server_port: int) -> WorkloadResult:
        """Run the long-context workload against the server.
        
        Args:
            server_port: Port of the LLM server
            
        Returns:
            WorkloadResult with prompt-processing tok/s
        """
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
                "stream": True,
                # Real llama-server only includes usage (with prompt_tokens)
                # in the stream when this is requested.
                "stream_options": {"include_usage": True}
            }
            
            ttft = None
            time_before_first_token = None
            prompt_tokens_processed = 0
            gen_tokens = 0
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
                        
                        chunk_time = time.time()
                        
                        # First, check for prompt_tokens in usage field (authoritative for total)
                        usage = chunk.get("usage", {})
                        if "prompt_tokens" in usage and prompt_tokens_processed == 0:
                            # This is the first chunk with usage, contains total prompt tokens
                            prompt_tokens_processed = usage["prompt_tokens"]
                        
                        # Track TTFT - first chunk with choices (content) is the first token
                        if ttft is None and chunk.get("choices"):
                            ttft = chunk_time - request_start_time
                            time_before_first_token = ttft
                        
                        # Count tokens - use usage field for accuracy
                        chunk_tokens = count_tokens_in_sse_chunk(chunk)
                        if chunk_tokens > 0:
                            if ttft is not None:
                                gen_tokens += chunk_tokens
                            # Note: prompt_tokens_processed is set above, not from this loop
                        
                        # Track time for prompt processing - stops at first content chunk
                        if ttft is None:
                            time_before_first_token = chunk_time - request_start_time
                    
                    # Calculate prompt tok/s
                    total_time = time.time() - request_start_time
                    # Use time_before_first_token as the prompt processing duration
                    actual_prompt_time = time_before_first_token or total_time
                    # Avoid division by zero
                    if actual_prompt_time < 0.001:
                        actual_prompt_time = 0.001
                    prompt_tok_s = prompt_tokens_processed / actual_prompt_time if prompt_tokens_processed > 0 else 0
                    
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


class BurstWorkloadRunner:
    """Burst workload: 4 concurrent requests, measures aggregate generation throughput."""
    
    DEFAULT_CONCURRENCY = 4
    DEFAULT_TOKENS_PER_REQUEST = 128
    
    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        tokens_per_request: int = DEFAULT_TOKENS_PER_REQUEST,
        server_port: Optional[int] = None  # Kept for backward compatibility
    ):
        self.concurrency = concurrency
        self.tokens_per_request = tokens_per_request
    
    async def run(self, server_port: int) -> WorkloadResult:
        """Run the burst workload with concurrent requests.
        
        Args:
            server_port: Port of the LLM server
            
        Returns:
            WorkloadResult with aggregate generation throughput
        """
        prompt_text = "Write a short story about AI."
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = [
                self._run_single_request(client, server_port, prompt_text, self.tokens_per_request)
                for _ in range(self.concurrency)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = [r for r in results if isinstance(r, tuple)]
            
            if not valid_results:
                return WorkloadResult(
                    workload_name="burst",
                    total_time=0,
                    tokens_per_second=0
                )
            
            # Calculate aggregate throughput
            times = [t for t, _, _ in valid_results if t is not None]
            total_time = max(times) if times else 0
            total_tokens = sum(tokens for _, tokens, _ in valid_results)
            
            # TTFT is the earliest time to first token across all requests
            ttft_values = [ttft for _, _, ttft in valid_results if ttft is not None]
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
        port: int,
        prompt_text: str,
        max_tokens: int
    ) -> Tuple[float, int, Optional[float]]:
        """Run a single request and return (elapsed_time, tokens, ttft)."""
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
                    
                    # Track TTFT
                    if ttft is None and chunk.get("choices"):
                        ttft = chunk_time - request_start_time
                    
                    # Count tokens
                    tokens_generated += count_tokens_in_sse_chunk(chunk)
                    
        except httpx.RequestError:
            pass
        
        elapsed = time.time() - request_start_time
        return elapsed, tokens_generated, ttft


# =============================================================================
# Ollama workload runners (native /api/generate endpoint)
# =============================================================================


class OllamaShortWorkloadRunner:
    """Short workload for ollama: ~512 tokens generated, measures peak generation tok/s and TTFT.
    
    Uses ollama's native /api/generate endpoint (streaming JSON chunks).
    """
    
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_PROMPT_TOKENS = 64
    
    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prompt_tokens: int = DEFAULT_PROMPT_TOKENS
    ):
        self.max_tokens = max_tokens
        self.prompt_tokens = prompt_tokens
    
    async def run(self, port: int) -> WorkloadResult:
        """Run the short workload against ollama's /api/generate endpoint.
        
        Args:
            port: Ollama server port (default 11434)
            
        Returns:
            WorkloadResult with peak generation tok/s and TTFT
        """
        # Create a prompt with approximately prompt_tokens tokens
        # Rough estimate: 1 token ≈ 4 characters
        prompt_char_count = self.prompt_tokens * 4
        prompt_text = "Once upon a time in a digital realm " * (prompt_char_count // 64)
        prompt_text = prompt_text[:prompt_char_count]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Ollama generate endpoint uses "prompt" instead of "messages"
            request = {
                "model": "benchmark-ollama-model",
                "prompt": prompt_text,
                "stream": True,
                "options": {
                    "num_predict": self.max_tokens,
                }
            }
            
            ttft = None
            tokens_generated = 0
            gen_rates: List[float] = []
            last_chunk_time = None
            window_start = None
            window_tokens = 0
            request_start_time = time.time()
            
            try:
                async with client.stream("POST", f"http://localhost:{port}/api/generate", json=request) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        chunk_time = time.time()
                        
                        # Track TTFT - first chunk with response content is the first token
                        if ttft is None and chunk.get("response"):
                            ttft = chunk_time - request_start_time
                            last_chunk_time = chunk_time
                        
                        # Count tokens in response field
                        response_text = chunk.get("response", "")
                        if response_text and ttft is not None:
                            tokens_in_chunk = len(response_text) // 4  # Rough estimate
                            if tokens_in_chunk > 0:
                                if window_start is None:
                                    window_start = chunk_time
                                    window_tokens = 0
                                window_tokens += tokens_in_chunk
                                # Coalesce bursts into rate windows (same logic as llama-server)
                                if chunk_time - window_start >= 0.05:  # GEN_WINDOW_SECONDS
                                    gen_rates.append(window_tokens / (chunk_time - window_start))
                                    window_start = chunk_time
                                    window_tokens = 0
                                last_chunk_time = chunk_time
                                tokens_generated += tokens_in_chunk
                        
                        # Stop when done
                        if chunk.get("done"):
                            break
                    
                    # Flush any open window
                    if window_start is not None and window_tokens > 0 and last_chunk_time:
                        window_duration = max(last_chunk_time - window_start, 0.05)
                        gen_rates.append(window_tokens / window_duration)
                    
                    peak_gen_tok_s = max(gen_rates) if gen_rates else 0
                    total_time = time.time() - request_start_time
                    
                    return WorkloadResult(
                        workload_name="short",
                        total_time=total_time,
                        tokens_per_second=peak_gen_tok_s,
                        ttft=ttft,
                        generated_tokens=tokens_generated
                    )
                    
            except httpx.RequestError:
                total_time = time.time() - request_start_time
                return WorkloadResult(
                    workload_name="short",
                    total_time=total_time,
                    tokens_per_second=0,
                    ttft=None,
                    generated_tokens=tokens_generated
                )


class OllamaLongContextWorkloadRunner:
    """Long-context workload for ollama: 16k token prompt, short generation.
    
    Measures prompt-processing tok/s for ollama's native API.
    """
    
    DEFAULT_PROMPT_TOKENS = 16384  # 16k tokens
    DEFAULT_GENERATION_TOKENS = 32
    
    def __init__(
        self,
        prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
        generation_tokens: int = DEFAULT_GENERATION_TOKENS
    ):
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
    
    async def run(self, port: int) -> WorkloadResult:
        """Run the long-context workload against ollama's /api/generate endpoint.
        
        Args:
            port: Ollama server port (default 11434)
            
        Returns:
            WorkloadResult with prompt-processing tok/s
        """
        # Create a prompt with approximately prompt_tokens tokens
        prompt_char_count = self.prompt_tokens * 4
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_char_count // 64)
        prompt_text = prompt_text[:prompt_char_count]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            request = {
                "model": "benchmark-ollama-model",
                "prompt": prompt_text,
                "stream": True,
                "options": {
                    "num_predict": self.generation_tokens,
                }
            }
            
            ttft = None
            time_before_first_token = None
            request_start_time = time.time()
            tokens_generated = 0
            
            try:
                async with client.stream("POST", f"http://localhost:{port}/api/generate", json=request) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        chunk_time = time.time()
                        
                        # Track TTFT - first chunk with response content
                        if ttft is None and chunk.get("response"):
                            ttft = chunk_time - request_start_time
                            time_before_first_token = ttft
                        
                        # Count tokens in response
                        response_text = chunk.get("response", "")
                        if response_text and ttft is not None:
                            tokens_in_chunk = len(response_text) // 4
                            if tokens_in_chunk > 0:
                                tokens_generated += tokens_in_chunk
                        
                        if chunk.get("done"):
                            break
                    
                    total_time = time.time() - request_start_time
                    actual_prompt_time = time_before_first_token or total_time
                    if actual_prompt_time < 0.001:
                        actual_prompt_time = 0.001
                    
                    # For long-context, prompt_tok_s = prompt_tokens / time_to_first_token
                    prompt_tok_s = self.prompt_tokens / actual_prompt_time if actual_prompt_time > 0 else 0
                    
                    return WorkloadResult(
                        workload_name="long-context",
                        total_time=total_time,
                        tokens_per_second=prompt_tok_s,
                        ttft=ttft,
                        prompt_tokens=self.prompt_tokens,
                        generated_tokens=tokens_generated
                    )
                    
            except httpx.RequestError:
                total_time = time.time() - request_start_time
                return WorkloadResult(
                    workload_name="long-context",
                    total_time=total_time,
                    tokens_per_second=0,
                    ttft=None,
                    prompt_tokens=self.prompt_tokens,
                    generated_tokens=tokens_generated
                )


class OllamaBurstWorkloadRunner:
    """Burst workload for ollama: concurrent requests to measure aggregate throughput.
    
    Uses multiple concurrent /api/generate requests to ollama.
    """
    
    DEFAULT_CONCURRENCY = 4
    DEFAULT_TOKENS_PER_REQUEST = 128
    
    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        tokens_per_request: int = DEFAULT_TOKENS_PER_REQUEST
    ):
        self.concurrency = concurrency
        self.tokens_per_request = tokens_per_request
    
    async def run(self, port: int) -> WorkloadResult:
        """Run the burst workload with concurrent ollama requests.
        
        Args:
            port: Ollama server port (default 11434)
            
        Returns:
            WorkloadResult with aggregate generation throughput
        """
        prompt_text = "Write a short story about AI."
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = [
                self._run_single_request(client, port, prompt_text, self.tokens_per_request)
                for _ in range(self.concurrency)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = [r for r in results if isinstance(r, tuple)]
            
            if not valid_results:
                return WorkloadResult(
                    workload_name="burst",
                    total_time=0,
                    tokens_per_second=0
                )
            
            times = [t for t, _, _ in valid_results if t is not None]
            total_time = max(times) if times else 0
            total_tokens = sum(tokens for _, tokens, _ in valid_results)
            
            # TTFT is the earliest time to first token across all requests
            ttft_values = [ttft for _, _, ttft in valid_results if ttft is not None]
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
        port: int,
        prompt_text: str,
        max_tokens: int
    ) -> Tuple[float, int, Optional[float]]:
        """Run a single ollama request and return (elapsed_time, tokens, ttft)."""
        request = {
            "model": "benchmark-ollama-model",
            "prompt": prompt_text,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
            }
        }
        
        ttft = None
        tokens_generated = 0
        request_start_time = time.time()
        
        try:
            async with client.stream("POST", f"http://localhost:{port}/api/generate", json=request) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    chunk_time = time.time()
                    
                    # Track TTFT - first chunk with response content
                    if ttft is None and chunk.get("response"):
                        ttft = chunk_time - request_start_time
                    
                    # Count tokens in response
                    response_text = chunk.get("response", "")
                    if response_text:
                        tokens_generated += len(response_text) // 4
                    
                    if chunk.get("done"):
                        break
                        
        except httpx.RequestError:
            pass
        
        elapsed = time.time() - request_start_time
        return elapsed, tokens_generated, ttft
