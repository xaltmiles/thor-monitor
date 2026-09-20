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
    
    # One shared BenchmarkRunner for the whole app: run state must survive
    # across requests, so never instantiate per-request.
    from ..benchmarks import BenchmarkRunner
    app.state.benchmark_runner = BenchmarkRunner()
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Render the main dashboard page."""
        latest = await get_latest_telemetry()
        history = await get_telemetry_history(limit=60)  # Last 60 samples (1 minute)
        
        # Get loaded models from probes
        from ..probes import run_probes
        probes_result = await run_probes()
        
        # Convert to list for template (Jinja2 has issues with dict in template context)
        history_list = list(history) if history else []
        
        content = await render_template("dashboard.html", {
            "latest": latest,
            "history": history_list,
            "models": probes_result,
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
    
    # Benchmark API endpoints
    @app.post("/api/benchmarks/start")
    async def api_benchmarks_start():
        """Start a new benchmark run."""
        runner = app.state.benchmark_runner
        try:
            run = await runner.trigger_run()
            return {
                "status": "started",
                "run": run.to_dict()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }, 500
    
    @app.get("/api/benchmarks/status")
    async def api_benchmarks_status(request: Request):
        """Get current benchmark run status."""
        runner = request.app.state.benchmark_runner
        run = await runner.get_status()
        
        if run:
            return {
                "status": run.state,
                "run": run.to_dict()
            }
        else:
            return {
                "status": "idle",
                "run": None
            }
    
    @app.get("/api/benchmarks/progress")
    async def api_benchmarks_progress(request: Request):
        """Get benchmark run progress."""
        runner = request.app.state.benchmark_runner
        run = await runner.get_status()
        
        if not run or run.state == "idle":
            return {
                "state": "idle",
                "active_workload": None,
                "eta": None,
                "workloads": []
            }
        
        # Get stored run data for progress tracking
        from ..store import get_benchmark_run
        stored = await get_benchmark_run(run.run_id)
        
        # Parse workload results if available
        workloads = []
        if stored and stored.get("workload_results"):
            import json
            try:
                results = json.loads(stored["workload_results"])
                for name, data in results.items():
                    workloads.append({
                        "name": data.get("workload_name", name),
                        "time": data.get("total_time", 0),
                        "tokens_per_second": data.get("tokens_per_second", 0)
                    })
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Calculate progress and ETA
        if run.state == "running" and workloads:
            # Estimate based on workload type
            total_workloads = 3  # short, long-context, burst
            completed = len(workloads)
            progress = completed / total_workloads
            
            # ETA based on average workload time (rough estimate)
            avg_time = sum(w["time"] for w in workloads) / completed if completed > 0 else 0
            remaining = total_workloads - completed
            eta = remaining * avg_time if avg_time > 0 else None
            
            # Active workload is the last one in progress
            active_workload = workloads[-1]["name"] if workloads else None
        else:
            progress = 0 if run.state == "queued" else 1
            eta = None
            active_workload = None
        
        return {
            "state": run.state,
            "active_workload": active_workload,
            "eta": eta,
            "workloads": workloads,
            "progress": progress
        }
    
    @app.get("/api/benchmarks/runs")
    async def api_benchmarks_runs(limit: int = 10):
        """Get recent benchmark runs."""
        from ..store import get_benchmark_runs
        
        runs = await get_benchmark_runs(limit=limit)
        return {"runs": runs}
    
    @app.get("/api/benchmarks/run/{run_id}")
    async def api_benchmarks_run(run_id: int):
        """Get a specific benchmark run."""
        from ..store import get_benchmark_run
        
        run = await get_benchmark_run(run_id)
        if run:
            return {"run": run}
        else:
            return {"error": "Run not found"}, 404
    
    return app


# Create and export the app instance
app = create_app()
