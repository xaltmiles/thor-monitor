"""Web module - FastAPI app and routes."""

import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..store import get_latest_telemetry, get_telemetry_history, get_telemetry_history_for_plots, get_telemetry_history_by_range
from .catalog import get_comparison_data, get_timeline_data
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
    
    @app.get("/catalog/comparison", response_class=HTMLResponse)
    async def catalog_comparison():
        """Render the comparison table page."""
        comparison_data = await get_comparison_data()
        content = await render_template("catalog_comparison.html", {
            "models": comparison_data["models"],
            "total_standard_runs": comparison_data["total_standard_runs"],
            "workload_types": comparison_data.get("workload_types", []),
        })
        return HTMLResponse(content=content)
    
    @app.get("/catalog/timeline", response_class=HTMLResponse)
    async def catalog_timeline():
        """Render the timeline page."""
        timeline_data = await get_timeline_data()
        content = await render_template("catalog_timeline.html", {
            "events": timeline_data["events"],
            "total_runs": timeline_data["total_runs"],
            "total_sessions": timeline_data["total_sessions"],
        })
        return HTMLResponse(content=content)
    
    @app.get("/plots", response_class=HTMLResponse)
    async def plots():
        """Render the plots page."""
        content = await render_template("plots.html", {})
        return HTMLResponse(content=content)
    
    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_plot(run_id: int):
        """Render a single benchmark run's timeline plot."""
        from ..store import get_benchmark_run
        
        run = await get_benchmark_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        content = await render_template("run_plot.html", {
            "run": run,
            "run_id": run_id,
        })
        return HTMLResponse(content=content)
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Render the main dashboard page."""
        latest = await get_latest_telemetry()
        history = await get_telemetry_history(limit=60)  # Last 60 samples (1 minute)
        
        # Get loaded models from probes
        from ..probes import run_probes
        probes_result = await run_probes()
        
        # Get warning thresholds from settings
        warning_gpu_temp = None
        try:
            from ..store import get_settings
            settings = await get_settings()
            if settings:
                warning_gpu_temp = settings.get("warning_gpu_temp")
        except Exception:
            # If settings can't be loaded, use defaults
            pass
        
        # Check if telemetry exceeds thresholds
        gpu_warning = None
        if latest:
            if warning_gpu_temp is not None and latest.get("gpu_temp") is not None:
                if latest["gpu_temp"] >= warning_gpu_temp:
                    gpu_warning = f"GPU temperature {latest['gpu_temp']}°C exceeds threshold {warning_gpu_temp}°C"
        
        # Convert to list for template (Jinja2 has issues with dict in template context)
        history_list = list(history) if history else []
        
        content = await render_template("dashboard.html", {
            "latest": latest,
            "history": history_list,
            "models": probes_result,
            "gpu_warning": gpu_warning,
            "warning_gpu_temp": warning_gpu_temp,
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
    
    @app.get("/api/plots/history")
    async def api_plots_history(limit: int = 60, since: str = None):
        """Get telemetry history for plots page.
        
        Returns only columns needed for plots (no process_memory, gpu_process_memory).
        Supports ?since=<timestamp> for incremental fetch.
        """
        if since:
            # Incremental fetch: get samples since the given timestamp
            return await get_telemetry_history_by_range(start_time=since, limit=limit)
        else:
            # Full fetch: get last N samples
            return await get_telemetry_history_for_plots(limit)
    
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
            "ollama_guidance": result.ollama_guidance,
        }
    
    @app.get("/api/probes/detect")
    async def api_probes_detect():
        """Run full probe detection."""
        result = await run_probes()
        return {
            "servers": [s.__dict__ for s in result.servers],
            "models": [m.__dict__ for m in result.models],
            "warning": result.warning,
            "ollama_guidance": result.ollama_guidance
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
            # FastAPI does not understand Flask-style (content, status) tuples:
            # those silently serialize as 200 responses. Raise instead.
            raise HTTPException(status_code=500, detail=str(e))
    
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
        WORKLOAD_ORDER = ["short", "long-context", "burst"]
        completed_names = [w["name"] for w in workloads]
        total_workloads = len(WORKLOAD_ORDER)
        
        if run.state == "running":
            progress = len(completed_names) / total_workloads
            
            # Active workload: first in fixed order that has not completed yet
            remaining_names = [n for n in WORKLOAD_ORDER if n not in completed_names]
            active_workload = remaining_names[0] if remaining_names else None
            
            # ETA based on average completed workload time (rough estimate)
            avg_time = sum(w["time"] for w in workloads) / len(workloads) if workloads else 0
            eta = len(remaining_names) * avg_time if avg_time > 0 else None
        elif run.state == "done":
            progress = 1
            eta = None
            active_workload = None
        else:
            progress = 0
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
        raise HTTPException(status_code=404, detail="Run not found")
    
    @app.get("/api/benchmarks/run/{run_id}/samples")
    async def api_benchmarks_run_samples(run_id: int):
        """Get samples for a specific benchmark run timeline."""
        from ..store import get_benchmark_run
        import json
        
        run = await get_benchmark_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Return samples_during as a list, parsing JSON if present
        samples_during = run.get("samples_during")
        if samples_during:
            try:
                samples = json.loads(samples_during)
            except (json.JSONDecodeError, TypeError):
                samples = []
        else:
            samples = []
        
        return {"samples": samples}
    
    @app.get("/api/benchmarks/run/{run_id}/timeline")
    async def api_benchmarks_run_timeline(run_id: int):
        """Get timeline data for a specific benchmark run."""
        from ..web.catalog import get_run_samples
        
        return await get_run_samples(run_id)
    
    @app.get("/api/settings")
    async def api_get_settings():
        """Get current settings."""
        from ..store import get_settings, DEFAULTS
        
        settings = await get_settings()
        if settings is None:
            # Return defaults if no settings exist (single source of truth: store.DEFAULTS)
            return dict(DEFAULTS)
        return settings
    
    @app.post("/api/settings")
    async def api_update_settings(request: Request):
        """Update settings."""
        from ..store import update_settings, DEFAULTS
        
        data = await request.json()
        
        # Validate sample_rate
        sample_rate = data.get("sample_rate")
        if sample_rate is not None:
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="sample_rate must be a positive number"
                )
        
        # Validate standard_workload_duration
        standard_workload_duration = data.get("standard_workload_duration")
        if standard_workload_duration is not None:
            if not isinstance(standard_workload_duration, (int, float)) or standard_workload_duration <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="standard_workload_duration must be a positive number"
                )
        
        # Validate max_queue_wait
        max_queue_wait = data.get("max_queue_wait")
        if max_queue_wait is not None:
            if not isinstance(max_queue_wait, (int, float)) or max_queue_wait <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="max_queue_wait must be a positive number"
                )
        
        # Validate warning_gpu_temp
        warning_gpu_temp = data.get("warning_gpu_temp")
        if warning_gpu_temp is not None:
            if not isinstance(warning_gpu_temp, (int, float)) or warning_gpu_temp < 0 or warning_gpu_temp > 200:
                raise HTTPException(
                    status_code=400,
                    detail="warning_gpu_temp must be a number between 0 and 200"
                )
        
        settings = await update_settings(
            sample_rate=sample_rate,
            standard_workload_duration=standard_workload_duration,
            max_queue_wait=max_queue_wait,
            warning_gpu_temp=warning_gpu_temp
        )
        
        # Update sampler interval if sample_rate changed (applies live)
        if sample_rate is not None and hasattr(app.state, "sampler"):
            app.state.sampler.set_interval(1.0 / sample_rate)
        
        return settings
    
    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page():
        """Render the settings page."""
        from ..store import get_settings
        
        # Get current settings for initial render
        try:
            settings = await get_settings()
        except Exception:
            settings = None
        
        content = await render_template("settings.html", {
            "settings": settings,
        })
        return HTMLResponse(content=content)
    
    return app


# Create and export the app instance
app = create_app()
