"""Fake Ollama server for testing - HTTP stub with streaming generate API."""

import asyncio
import json
from datetime import datetime, timezone
from aiohttp import web
from typing import Dict, Optional


class FakeOllamaServer:
    """Fake ollama server with streaming /api/generate endpoint."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 11435):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Model name returned by /api/ps
        self._model_name = "fake-ollama-model"
        
        # Active requests counter (for queue-until-idle testing)
        self._active_requests = 0
        
        # Streaming response configuration
        self._streaming_config = {
            "delay_ms": 50,  # delay between chunks for realistic pacing
            "first_chunk_delay_ms": 200,  # delay for first chunk (TTFT simulation)
            "tokens_per_chunk": 8,  # tokens per SSE chunk
        }
        
        # Setup routes
        self.app.router.add_get("/api/ps", self.handle_ps)
        self.app.router.add_post("/api/generate", self.handle_generate)
        self.app.router.add_post("/api/chat", self.handle_chat)
    
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
    
    async def handle_ps(self, request: web.Request) -> web.Response:
        """Handle /api/ps endpoint - return list of loaded models."""
        # Simulate ollama's /api/ps response
        # When debug logging is enabled, models have "details" with "streaming" field
        response = {
            "models": [
                {
                    "name": self._model_name,
                    "model": self._model_name,
                    "modified_at": "2026-09-20T12:00:00.000Z",
                    "size": 4600000000,
                    "digest": "sha256:fake123",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M",
                        "streaming": True,  # Indicates debug logging enabled
                    },
                }
            ]
        }
        return web.json_response(response)
    
    async def handle_generate(self, request: web.Request) -> web.Response:
        """Handle /api/generate endpoint - streaming generation."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400
            )
        
        model = body.get("model", self._model_name)
        prompt = body.get("prompt", "")
        stream = body.get("stream", True)
        options = body.get("options", {})
        max_tokens = options.get("num_predict", 512)
        
        if not stream:
            return web.json_response(
                {"error": "Non-streaming not supported"},
                status=501
            )
        
        # Create SSE stream response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)
        
        try:
            # Simulate TTFT
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
                
                # Build ollama-style chunk
                finish_reason = "stop" if tokens_remaining == 0 else None
                chunk = {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": "chunk " if chunk_id == 0 else f"token_{chunk_id} ",
                    "done": finish_reason is not None,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 64,
                    "prompt_eval_duration": 0,
                    "eval_count": chunk_id + 1,
                    "eval_duration": 0,
                }
                
                if finish_reason:
                    chunk["done"] = True
                    chunk["done_reason"] = "stop"
                
                # Send NDJSON chunk
                data = json.dumps(chunk) + "\n"
                await response.write(data.encode("utf-8"))
                await response.drain()
                
                # Simulate token pacing
                if tokens_remaining > 0:
                    await asyncio.sleep(chunk_delay)
                
                chunk_id += 1
            
        except Exception:
            pass
        finally:
            await response.write_eof()
        
        return response
    
    async def handle_chat(self, request: web.Request) -> web.Response:
        """Handle /api/chat endpoint - streaming chat."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400
            )
        
        model = body.get("model", self._model_name)
        messages = body.get("messages", [])
        stream = body.get("stream", True)
        
        if not stream:
            return web.json_response(
                {"error": "Non-streaming not supported"},
                status=501
            )
        
        # Create SSE stream response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)
        
        try:
            first_chunk_delay = self._streaming_config.get("first_chunk_delay_ms", 200) / 1000.0
            await asyncio.sleep(first_chunk_delay)
            
            # For simplicity, generate a fixed number of tokens
            max_tokens = 128
            tokens_per_chunk = self._streaming_config.get("tokens_per_chunk", 8)
            chunk_delay = self._streaming_config.get("delay_ms", 50) / 1000.0
            
            chunk_id = 0
            tokens_remaining = max_tokens
            
            while tokens_remaining > 0:
                current_chunk_tokens = min(tokens_per_chunk, tokens_remaining)
                tokens_remaining -= current_chunk_tokens
                
                finish_reason = "stop" if tokens_remaining == 0 else None
                chunk = {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": "chunk " if chunk_id == 0 else f"token_{chunk_id} "
                    },
                    "done": finish_reason is not None,
                }
                
                data = json.dumps(chunk) + "\n"
                await response.write(data.encode("utf-8"))
                await response.drain()
                
                if tokens_remaining > 0:
                    await asyncio.sleep(chunk_delay)
                
                chunk_id += 1
            
        except Exception:
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
    
    def set_model_name(self, name: str):
        """Set the model name returned by /api/ps."""
        self._model_name = name
    
    def set_streaming_config(self, config: Dict[str, int]):
        """Configure streaming behavior for testing."""
        self._streaming_config.update(config)
