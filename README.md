# Thor Monitor

A self-hosted web dashboard on your nvidia-thor PC that answers:

- **Which LLM server** is running (ollama or unsloth studio with llama.cpp)?
- **Which model** is currently loaded?
- **At what speed** does it serve?

It detects the running server and loaded model by probing—never by configuration—and never loads or changes models itself.

---

## Features

- **Probe-based detection**: Discovers which server (ollama or llama-server) is running and which model is loaded, purely by inspecting processes, APIs, and system tools
- **Live telemetry**: GPU utilization, temperature, power, per-process GPU memory, unified memory totals
  - **Unified memory** is measured from `/proc/meminfo`: `MemTotal` (total system RAM) and `MemAvailable` (free + cacheable). Used = Total − Available.
    - `tegrastats` uses the same kernel accounting for its RAM line, so its total matches `MemTotal`.
    - `nvidia-smi` reports per-process GPU-side allocations (e.g., `llama-server 53132MiB`), not system-wide RAM usage.
- **Passive observation**: Measures actual tok/s during your usage without generating load
- **Benchmark runs**: Standard suite (short, long-context, burst) with results stored in a persistent catalog
- **Catalog**: Historical record of all benchmark runs and passive sessions for model comparisons

---

## Requirements

- Python 3.13 via `uv`
- Jetson device (Jetson Thor, Orin Nano, etc.) with:
  - `nvidia-smi` for GPU telemetry
  - CUDA 12.6+ (JetPack 6.1+)
- One or both of:
  - **ollama** (v0.32+): for `ollama serve`
  - **llama-server** (with Prometheus metrics enabled): for `llama-server --parallel 4 --metrics`

---

## Quick Start

### Manual run from the repo

```bash
# Install dependencies
uv sync

# Start the monitor
uv run monitor
```

The dashboard will be available at `http://<your-thor-ip>:8123`

### Running in LAN mode

The app binds to `0.0.0.0:8123` by default, making it reachable from any device on your LAN. Access it from your laptop/phone at:

```
http://<your-thor-ip>:8123
```

Replace `<your-thor-ip>` with the IP address of your Jetson Thor (e.g., `192.168.1.100`).

---

## Setup

### 1. Enable ollama debug logging (one-time, required for passive tok/s)

To see passive token-generation speed when using ollama, enable debug logging:

```bash
# Edit the ollama service
sudo systemctl edit ollama

# Add this override:
[Service]
Environment="OLLAMA_DEBUG=1"

# Reload and restart
sudo systemctl daemon-reexec
sudo systemctl restart ollama
```

### 2. Start llama-server with metrics (if using llama.cpp)

```bash
llama-server --parallel 4 --metrics
```

The metrics endpoint at `/metrics` is required for passive tok/s observation.

---

## Smoke Checklist

Run this checklist after starting the monitor to verify everything works:

### 1. Dashboard starts and responds

```bash
# From a laptop or phone on the same LAN
curl http://<your-thor-ip>:8123/
```

Expected: HTML with "Monitor Dashboard" title and live telemetry cards.

### 2. Detect running server and model

```bash
curl http://<your-thor-ip>:8123/api/probes/detect
```

Expected: JSON with `servers` array (containing `ollama` or `llama-server`) and `models` array with model info.

### 3. Telemetry is flowing

```bash
curl http://<your-thor-ip>:8123/api/telemetry/latest
```

Expected: JSON with `memory_total`, `memory_used`, `gpu_util`, `gpu_temp`, `gpu_power`.

### 4. Run a benchmark

```bash
curl -X POST http://<your-thor-ip>:8123/api/benchmarks/start
```

Expected: JSON with `status: "started"` and a run ID.

Check progress:

```bash
curl http://<your-thor-ip>:8123/api/benchmarks/progress
```

Expected: `state: "running"` during benchmark, then `state: "done"`.

### 5. Catalog entry appears

After the benchmark completes, check the catalog:

```bash
curl http://<your-thor-ip>:8123/api/benchmarks/runs
```

Expected: JSON with `runs` array containing the benchmark result with `tokens_per_second` and workload results.

---

## systemd unit file (for always-on mode)

A `systemd` unit file is provided for deploying the monitor as a persistent service:

**File**: `monitor.unicorn.service`

### Installation (copy-paste, not auto-installed)

```bash
# Copy the unit file
sudo cp monitor.unicorn.service /etc/systemd/system/

# Edit to match your environment (User, Group, WorkingDirectory)
sudoedit /etc/systemd/system/monitor.unicorn.service

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable monitor.unicorn
sudo systemctl start monitor.unicorn

# Check status
sudo systemctl status monitor.unicorn
sudo journalctl -u monitor.unicorn -f
```

### Unit file contents

See [`monitor.unicorn.service`](monitor.unicorn.service) for the full unit file. Key configuration:

- Runs as non-root (`User=eqr`, `Group=eqr`)
- Working directory: your monitor repo
- Restart policy: `on-failure` with 10s delay
- Logs to systemd journal (`StandardOutput=journal`)

---

## Catalog: How to read it

The **Catalog** is the accumulated history of all benchmark runs and passive sessions, stored in `~/.monitor/monitor.db`.

### What's stored

| Entity | Description |
|--------|-------------|
| `models` | Identity + metadata (name, quant, context length, file size, server type) |
| `benchmark_runs` | Results from each benchmark run (workload results, tok/s, memory/GPU footprint) |
| `sessions` | Periods of passive observation (start/end times, observed tok/s) |
| `telemetry_samples` | Raw GPU/memory samples (59,999 rows retained, oldest purged) |

### Comparison table

Visit `/catalog/comparison` to see a per-model table with:

- Best/avg tok/s across standard runs (short, long-context, burst)
- Memory footprint (file size + GPU usage during runs)
- Number of standard runs

**Note**: Only *standard* runs (default suite parameters) appear in the comparison table. *Custom* runs (modified parameters) are tagged and stored separately.

### Timeline

Visit `/catalog/timeline` to see events in chronological order:

- Benchmark runs (colored by workload: short, long-context, burst)
- Passive sessions (gaps in actual usage)
- Server/model changes (detected via probes)

Use the timeline to spot patterns (e.g., thermal throttling over time).

---

## Troubleshooting

### No telemetry samples

- Check `nvidia-smi` is accessible: `nvidia-smi --query-compute-apps`
- Verify telemetry sources are properly installed: `uv pip list | grep -i nvidia`
- Check logs: `journalctl -u monitor.unicorn` (if using systemd)

### Passive tok/s not showing

- For **ollama**: ensure debug logging is enabled (see setup above)
- For **llama-server**: ensure metrics endpoint is enabled (`--metrics` flag)

### Server not detected

- Verify the server is running: `ps aux | grep -E 'ollama|llama-server'`
- Check probe API: `curl http://localhost:8123/api/probes/detect`

### Port 8123 already in use

```bash
# Find what's using the port
sudo lsof -i :8123
```

---

## Development

### Run tests

```bash
uv run pytest -q
```

### Run smoke test

```bash
make smoke
```

This starts the real app, drives real HTTP and the real SQLite store, and fails on any break that fixtures mask (unwired sources, broken imports, dead real-system paths).

### Add a new server type

1. Add process detection to `probes/server.py`
2. Add model detection to `probes/model.py`
3. Add HTTP endpoints in `web/routes.py` (if needed)

---

## License

MIT
