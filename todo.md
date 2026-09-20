# Implementation Plan for Issue #2: Walking Skeleton

## Requirements from Issue #2:
1. `uv run monitor` serves a dashboard over HTTP, reachable from the LAN
2. First run creates the SQLite store with a telemetry-samples table
3. Unified-memory totals sampled at 1 Hz via the telemetry-source interface (no direct system calls at call sites)
4. Dashboard shows live-updating memory values
5. Test suite: fixture-fed telemetry source → assertions on HTTP responses and database rows
6. No config file is created; all runtime state lives in the store

## Breaking down the Implementation:

### 1. Project Structure
- `monitor/` package
  - `__init__.py` - Package init
  - `main.py` - Entry point (cli entrypoint)
  - `app.py` - FastAPI app setup
  - `store.py` - SQLite database operations
  - `telemetry/` - Telemetry sources module
    - `__init__.py`
    - `interface.py` - Telemetry source interface (abstraction layer)
    - `memory.py` - Memory sampling implementation
    - `process.py` - Process table sampling (for detection)
    - `nvidia.py` - nvidia-smi sampling (for GPU stats)
  - `sampler.py` - 1 Hz sampling loop
  - `server.py` - HTTP server setup
  - `web/` - Web UI module
    - `__init__.py`
    - `routes.py` - HTTP routes
    - `templates/` - HTML templates
  - `fixtures/` - Test fixtures

### 2. Key Components:

#### Store (SQLite)
- Create database on first run
- `telemetry_samples` table: id, timestamp, memory_total, memory_free, memory_used, gpu_util, gpu_temp, gpu_power, process_memory
- Other tables for models, benchmark runs, sessions, settings (from parent spec)

#### Telemetry Interface
- Abstract interface for telemetry sources
- Fixture-fed for testing
- Real implementations for memory, GPU, process data

#### Sampler
- Runs at 1 Hz
- Collects from telemetry sources
- Writes to database

#### Web UI
- FastAPI + Jinja + htmx
- Dashboard page with live-updating memory values
- LAN-bound (0.0.0.0)

#### Tests
- pytest
- HTTP endpoint testing
- Database row assertions
- Fixture-fed telemetry sources

### 3. Dependencies to Add:
- fastapi
- uvicorn
- jinja2
- htmx
- aiosqlite (async SQLite)
- psutil (process info)
- pydantic (data validation)

Let me implement step by step.
