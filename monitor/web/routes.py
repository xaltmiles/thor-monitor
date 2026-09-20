"""Web module - FastAPI app and routes."""

import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..store import get_latest_telemetry, get_telemetry_history
from ..telemetry.real import RealMemorySource, RealGPUSource, RealProcessSource
from ..telemetry.fixtures import FixtureMemorySource, FixtureGPUSource, FixtureProcessSource, FixtureGPUMemorySource, FixtureLLaMAStatsSource, FixtureTelemetrySource
from ..sampler import Sampler
from ..probes import run_probes, FixtureServerDetector, FixtureModelDetector, ServerDetectorImpl
from ..probes.interface import ProbeResult
import jinja2

# Setup templates directory
templates_dir = Path(__file__).parent / "templates"
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(templates_dir)),
    autoescape=True,
    enable_async=True,
)
jinja_env.cache = None  # Disable caching


def _from_json(value):
    """Parse a JSON string (e.g. gpu_process_memory from the store)."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return []


jinja_env.filters["from_json"] = _from_json


async def render_template(template_name: str, context: dict):
    """Render a template with the given context."""
    template = jinja_env.get_template(template_name)
    return await template.render_async(**context)


def create_app():
    app = FastAPI(title="Monitor")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Render the main dashboard page."""
        latest = await get_latest_telemetry()
        history = await get_telemetry_history(limit=60)  # Last 60 samples (1 minute)
        
        # Convert to list for template (Jinja2 has issues with dict in template context)
        history_list = list(history) if history else []
        
        content = await render_template("dashboard.html", {
            "latest": latest,
            "history": history_list,
        })
        return HTMLResponse(content=content)
    
    @app.get("/api/telemetry/latest")
    async def api_telemetry_latest():
        """Get the latest telemetry sample."""
        return await get_latest_telemetry()
    
    @app.get("/api/telemetry/history")
    async def api_telemetry_history(limit: int = 60):
        """Get recent telemetry history."""
        return await get_telemetry_history(limit)
    
    @app.get("/api/fixtures/telemetry")
    async def api_fixture_telemetry():
        """Get fixture telemetry for testing."""
        fixture = FixtureTelemetrySource(
            memory_source=FixtureMemorySource(total=32_000_000_000, free=16_000_000_000, used=16_000_000_000),
            gpu_source=FixtureGPUSource(util=45.0, temp=70.0, power=150.0),
            gpu_memory_source=FixtureGPUMemorySource(processes=[
                {"pid": 1852, "name": "ollama", "gpu_memory": 8_000_000_000},
                {"pid": 8816, "name": "llama-server", "gpu_memory": 12_000_000_000},
            ]),
            llama_stats_source=FixtureLLaMAStatsSource(
                prompt_tokens=10000,
                generated_tokens=5000,
                speculative_accepts=500,
                prompt_tokens_rate=10.0,
                generated_tokens_rate=5.0,
                speculative_accepts_rate=0.5
            ),
            process_source=FixtureProcessSource(processes=[])  # Empty process list for simplicity
        )
        return await fixture.collect()
    
    # Probe endpoints
    @app.get("/api/probes/servers")
    async def api_probes_servers():
        """Get detected servers."""
        detector = ServerDetectorImpl()
        servers = await detector.detect()
        return {"servers": [s.__dict__ for s in servers]}
    
    @app.get("/api/probes/models")
    async def api_probes_models():
        """Get loaded models from really detected servers."""
        result = await run_probes()
        return {
            "models": [m.__dict__ for m in result.models],
            "warning": result.warning,
        }
    
    @app.get("/api/probes/detect")
    async def api_probes_detect():
        """Run full probe detection."""
        result = await run_probes()
        return {
            "servers": [s.__dict__ for s in result.servers],
            "models": [m.__dict__ for m in result.models],
            "warning": result.warning
        }
    
    @app.get("/api/fixtures/probes")
    async def api_fixture_probes():
        """Get fixture probe data for testing."""
        return {
            "servers": [
                {"pid": 1234, "name": "ollama", "type": "ollama", "port": 11434, "cmdline": ["ollama", "serve"]},
                {"pid": 5678, "name": "llama-server", "type": "llama-server", "port": 8080, "cmdline": ["llama-server", "--model", "/path/to/model.gguf"]},
            ],
            "models": [
                {"name": "llama3", "quant": "8b", "context_length": 8192, "file_size": 4600000000, "server_type": "ollama"},
                {"name": "mistral", "quant": "Q4_K_M", "context_length": 8192, "file_size": 4200000000, "server_type": "llama-server"},
            ],
            "warning": "Multiple loaded models detected: llama3, mistral"
        }
    
    @app.get("/api/telemetry/latest")
    async def api_telemetry_latest():
        """Get the latest telemetry sample."""
        return await get_latest_telemetry()
    
    @app.get("/api/telemetry/history")
    async def api_telemetry_history(limit: int = 60):
        """Get recent telemetry history."""
        return await get_telemetry_history(limit)
    
    @app.get("/api/fixtures/llama-stats")
    async def api_fixture_llama_stats():
        """Get fixture llama-server stats for testing."""
        return {
            "prompt_tokens": 10000,
            "generated_tokens": 5000,
            "speculative_accepts": 500,
            "prompt_tokens_rate": 10.0,
            "generated_tokens_rate": 5.0,
            "speculative_accepts_rate": 0.5
        }
    
    return app


# Create and export the app instance
app = create_app()
