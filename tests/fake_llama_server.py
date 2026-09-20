"""Fake LLaMA server for testing - HTTP stub with programmable metrics."""

import asyncio
from aiohttp import web
import re
from typing import Dict, Optional


class FakeLLaMAServer:
    """Fake llama-server with Prometheus /metrics endpoint."""
    
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
        
        # Setup routes
        self.app.router.add_get("/metrics", self.handle_metrics)
        self.app.router.add_post("/metrics/reset", self.handle_reset)
        self.app.router.add_post("/metrics/inc", self.handle_increment)
    
    @property
    def url(self) -> str:
        """Return the base URL of the fake server."""
        return f"http://{self.host}:{self.port}"
    
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
        }
        lines = []
        for key, metric in self.METRIC_NAMES.items():
            lines.append(f"# HELP {metric.replace(':', '_')} {helps[key]}")
            lines.append(f"# TYPE {metric.replace(':', '_')} counter")
            lines.append(f"{metric} {self._prom_value(self._counters[key])}")
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
