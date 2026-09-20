# Implementation Plan for Ticket #7

## Current State
- Ticket #6 implemented: Benchmark Run core with short workload (trigger, queue-until-idle, TTFT + peak tok/s)
- Benchmark runner exists in `monitor/benchmarks.py`
- Fake LLaMA server supports streaming with configurable parameters
- Database schema supports benchmark_runs table with all required fields

## What #7 Adds
1. **Long-context workload**: 16k-token prompt, short generation → prompt-processing tok/s
2. **Burst workload**: 4 concurrent requests → aggregate generation throughput
3. **Progress display**: Active workload name + ETA
4. **Suite parameters**: Settings defaults + standard vs custom tagging

## Files to Modify

### 1. monitor/benchmarks.py
- Add `LongContextWorkloadRunner` class
- Add `BurstWorkloadRunner` class  
- Add `SuiteSettings` for defaults and custom tagging
- Modify `_run_benchmark` to run all three workloads
- Add progress tracking endpoint

### 2. monitor/web/routes.py
- Add `/api/benchmarks/progress` endpoint

### 3. monitor/web/templates/dashboard.html
- Add progress display showing active workload and ETA

### 4. tests/test_benchmarks.py
- Add tests for long-context workload
- Add tests for burst workload (concurrent requests)
- Add tests for progress tracking

### 5. tests/fake_llama_server.py
- Support configurable request duration (for realistic burst testing)
- Support concurrent request simulation

## Suite Parameters (from spec)
- Long-context: 16k token prompt, 32 token generation
- Burst: 4 concurrent requests
- Short: current 64 prompt, 512 tokens

## Tagging Logic
- Standard run: all workloads use default parameters
- Custom run: any workload deviates from defaults

## Progress Display
- Show which workload is executing
- Estimate ETA based on previous workload times
