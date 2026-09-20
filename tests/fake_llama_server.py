"""Fake LLaMA server for testing - HTTP stub with programmable metrics."""

import asyncio
import json
import time
from aiohttp import web
import re
from typing import Dict, Optional, List


class FakeLLaMAServer:
    """Fake llama-server with Prometheus /metrics endpoint and streaming OpenAI /v1/chat/completions."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 18080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Initialize metrics counters (API keys map to real llama.cpp
        # Prometheus names in _format_metrics)
        self._counters = {
            "prompt_tokens_total": 0,
            "generated_tokens_total": 0,
            "speculative_accepts_total": 0,
        }
        
        # Active requests counter (for queue-until-idle testing)
        self._active_requests = 0
        
        # OpenAI streaming response configuration
        self._streaming_config = {
            "delay_ms": 50,  # delay between chunks for realistic pacing
            "first_chunk_delay_ms": 200,  # delay for first chunk (TTFT simulation)
            "tokens_per_chunk": 8,  # tokens per SSE chunk
        }
        
        # Setup routes
        self.app.router.add_get("/metrics", self.handle_metrics)
        self.app.router.add_post("/metrics/reset", self.handle_reset)
        self.app.router.add_post("/metrics/inc", self.handle_increment)
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
    
    @property
    def url(self) -> str:
        """Return the base URL of the fake server."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def active_requests(self) -> int:
        """Get current active requests count."""
        return self._active_requests
    
    @active_requests.setter
    def active_requests(self, value: int):
        """Set active requests count (for testing queue behavior)."""
        self._active_requests = max(0, value)
    
    # API counter key -> real llama.cpp Prometheus metric name
    METRIC_NAMES = {
        "prompt_tokens_total": "llamacpp:prompt_tokens_total",
        "generated_tokens_total": "llamacpp:tokens_predicted_total",
        "speculative_accepts_total": "llamacpp:spec_decode_num_accepted_tokens_total",
    }
    
    @staticmethod
    def _prom_value(value: int) -> str:
        """Render a counter like Prometheus does: large values in scientific notation."""
        if value >= 100_000:
            return f"{value:.5e}"
        return str(value)
    
    def _format_metrics(self) -> str:
        """Format metrics in Prometheus text format using real llama.cpp names."""
        helps = {
            "prompt_tokens_total": "Total number of prompt tokens processed",
            "generated_tokens_total": "Total number of tokens predicted (generated)",
            "speculative_accepts_total": "Total number of speculative draft tokens accepted",
            "requests_processing": "Number of requests processing",
        }
        lines = []
        for key, metric in self.METRIC_NAMES.items():
            lines.append(f"# HELP {metric.replace(':', '_')} {helps[key]}")
            lines.append(f"# TYPE {metric.replace(':', '_')} counter")
            lines.append(f"{metric} {self._prom_value(self._counters[key])}")
        # Add active requests as a gauge
        lines.append("# HELP llamacpp:requests_processing Number of requests processing")
        lines.append("# TYPE llamacpp:requests_processing gauge")
        lines.append(f"llamacpp:requests_processing {self._active_requests}")
        return "\n".join(lines) + "\n"
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Handle /metrics endpoint."""
        return web.Response(
            text=self._format_metrics(),
            content_type="text/plain"
        )
    
    async def handle_reset(self, request: web.Request) -> web.Response:
        """Reset all metrics counters."""
        data = await request.json()
        counters = data.get("counters", {})
        
        for key in self._counters:
            if key in counters:
                self._counters[key] = counters[key]
        
        return web.json_response({"status": "ok", "counters": self._counters})
    
    async def handle_increment(self, request: web.Request) -> web.Response:
        """Increment metrics counters."""
        data = await request.json()
        increments = data.get("increments", {})
        
        for key, value in increments.items():
            if key in self._counters:
                self._counters[key] += value
        
        return web.json_response({"status": "ok", "counters": self._counters})
    
    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle OpenAI-compatible streaming /v1/chat/completions endpoint."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400
            )
        
        # Extract configuration
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        stream = body.get("stream", True)
        
        if not stream:
            # Non-streaming not implemented
            return web.json_response(
                {"error": "Non-streaming not supported"},
                status=501
            )
        
        # Count tokens in prompt
        prompt_text = "".join(msg.get("content", "") for msg in messages if msg.get("role") == "user")
        prompt_token_count = len(prompt_text.split()) * 1.3  # rough estimate
        
        # Update metrics
        self._counters["prompt_tokens_total"] += int(prompt_token_count)
        self._counters["generated_tokens_total"] += max_tokens
        
        # Create SSE stream response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)
        
        try:
            # Simulate TTFT (time to first token)
            first_chunk_delay = self._streaming_config.get("first_chunk_delay_ms", 200) / 1000.0
            await asyncio.sleep(first_chunk_delay)
            
            # Generate response chunks
            tokens_remaining = max_tokens
            tokens_per_chunk = self._streaming_config.get("tokens_per_chunk", 8)
            chunk_delay = self._streaming_config.get("delay_ms", 50) / 1000.0
            
            chunk_id = 0
            while tokens_remaining > 0:
                current_chunk_tokens = min(tokens_per_chunk, tokens_remaining)
                tokens_remaining -= current_chunk_tokens
                
                # Build SSE message
                finish_reason = "length" if tokens_remaining == 0 else None
                chunk = {
                    "id": f"chatcmpl-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "fake-model",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": "chunk " if chunk_id == 0 else f"token_{chunk_id} ",
                            "role": "assistant"
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": int(prompt_token_count),
                        "completion_tokens": max_tokens,
                        "total_tokens": int(prompt_token_count) + max_tokens
                    }
                }
                
                # Send SSE chunk
                data = f"data: {json.dumps(chunk)}\n\n"
                await response.write(data.encode("utf-8"))
                await response.write(b"\n")  # Extra newline for safety
                await response.drain()
                
                # Simulate token pacing
                if tokens_remaining > 0:
                    await asyncio.sleep(chunk_delay)
                
                chunk_id += 1
            
            # Send [DONE] message
            done_chunk = {"id": f"chatcmpl-{chunk_id}", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake-model", "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
            data = f"data: {json.dumps(done_chunk)}\n\n"
            await response.write(data.encode("utf-8"))
            await response.drain()
            
        except Exception:
            # Client may disconnect; ignore errors during stream
            pass
        finally:
            await response.write_eof()
        
        return response
    
    async def start(self):
        """Start the fake server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
    
    async def stop(self):
        """Stop the fake server."""
        if self.runner:
            await self.runner.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    def set_counters(self, counters: Dict[str, int]):
        """Set metric counters directly."""
        for key, value in counters.items():
            if key in self._counters:
                self._counters[key] = value
    
    def increment(self, key: str, value: int = 1):
        """Increment a metric counter."""
        if key in self._counters:
            self._counters[key] += value
    
    def set_streaming_config(self, config: Dict[str, int]):
        """Configure streaming behavior for testing."""
        self._streaming_config.update(config)
