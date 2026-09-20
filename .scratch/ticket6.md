# Ticket #6 Implementation Plan

## What to Build

A benchmark run system that:
1. Triggered via dashboard button
2. Queues until server idle (checks active requests, 2-min timeout)
3. Runs short workload (small prompt, ~512 tokens)
4. Captures TTFT and peak generation tok/s
5. Samples memory/GPU before/during/after
6. Stores run with model identity, server type, workload params, "standard" tag

## Files to Create/Modify

### New Files:
1. `monitor/benchmarks.py` - BenchmarkRunner class with state machine
2. `monitor/benchmarks/store.py` or extend `monitor/store.py` - insert/get benchmark run functions
3. `tests/test_benchmarks.py` - unit tests
4. `tests/test_benchmark_e2e.py` - e2e test with fake server

### Modify Files:
1. `monitor/store.py` - add `insert_benchmark_run`, `update_benchmark_run`, `get_benchmark_runs`, etc.
2. `monitor/web/routes.py` - add `/api/benchmarks/start` and `/api/benchmarks/status` endpoints
3. `monitor/web/templates/dashboard.html` - add Benchmark card with trigger button
4. `monitor/main.py` - wire in BenchmarkRunner
5. `tests/fake_llama_server.py` - add streaming /v1/chat/completions endpoint + active requests gauge

## Design Decisions

1. **BenchmarkRunner state machine**: queued → running → done | aborted(reason)
2. **Queue-until-idle**: poll `llamacpp:requests_processing` gauge every 2s, timeout at 120s
3. **Workload**: OpenAI-compatible streaming POST, max_tokens=512
4. **TTFT**: time to first SSE chunk from response
5. **Peak tok/s**: max over inter-chunk deltas (tokens per chunk from usage field)
6. **Footprint sampling**: use RealMemorySource + RealGPUSource + RealGPUMemorySource
7. **Model identity**: from probes (ModelDetectorImpl over detected servers)
8. **Store format**: JSON TEXT columns for samples (consistent with existing pattern)

## Implementation Steps

1. Extend fake_llama_server.py with streaming endpoint and active requests control
2. Add store functions for benchmark_runs
3. Implement BenchmarkRunner with state machine
4. Add API routes for start/status
5. Add dashboard UI
6. Wire into main.py
7. Write tests (unit + e2e)
8. Run make test && make smoke
