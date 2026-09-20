"""Web module - FastAPI app and routes."""

import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..store import get_latest_telemetry, get_telemetry_history
from ..telemetry import RealMemorySource, RealGPUSource, RealProcessSource
from ..telemetry.fixtures import (
    FixtureMemorySource, FixtureGPUSource, FixtureProcessSource, FixtureTelemetrySource
)
from ..sampler import Sampler
from pathlib import Path
import jinja2

# Create a new app instance to avoid cache issues
def create_app():
    app = FastAPI(title="Monitor")
    
    # Setup templates directory with custom Jinja2 environment
    templates_dir = Path(__file__).parent / "templates"
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=True,
        enable_async=True,
    )
    jinja_env.cache = None  # Disable caching
    
    async def render_template(template_name: str, context: dict):
        """Render a template with the given context."""
        template = jinja_env.get_template(template_name)
        return await template.render_async(**context)
    
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
            process_source=FixtureProcessSource(processes=[])  # Empty process list for simplicity
        )
        return await fixture.collect()
    
    return app

app = create_app()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard page."""
    latest = await get_latest_telemetry()
    history = await get_telemetry_history(limit=60)  # Last 60 samples (1 minute)
    
    # Convert to list for template (Jinja2 has issues with dict in template context)
    history_list = list(history) if history else []
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "latest": latest,
            "history": history_list,
        }
    )


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
        process_source=FixtureProcessSource(processes=[])
    )
    return await fixture.collect()
