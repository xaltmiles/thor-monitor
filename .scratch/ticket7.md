# Ticket #7 Implementation Plan

## Parent
- Spec: xaltmiles/thor-monitor#1

## What to build

The Benchmark Run executes the full standard suite: the long-context workload (16k-token prompt, short generation → prompt-processing tok/s) and the burst workload (4 concurrent requests → aggregate throughput), in addition to the short workload. Run progress shows which workload is executing and rough ETA.

## Acceptance criteria

- [x] Long-context workload runs and records prompt-processing tok/s
- [x] Burst workload runs 4 concurrent requests and records aggregate generation throughput
- [x] A full run produces one stored record with all three workloads' results
- [x] Progress display names the active workload and ETA
- [x] Suite parameters come from settings defaults; deviations tag the run custom (tagging enforced even though the settings page arrives later)
- [x] Tests cover all three workloads against the fake LLM server, including concurrency

## Blocked by

[T5] Benchmark Run core.

## Current state

Ticket #6 implemented:
- Benchmark Run core: trigger, queue-until-idle, short workload
- TTFT and peak generation tok/s capture
- Storage with model identity, server type, workload params
- Fake LLM server with streaming endpoint and active requests gauge
- Unit and e2e tests

## What's missing for #7

1. **Long-context workload**: 16k-token prompt, short generation → record prompt-processing tok/s
2. **Burst workload**: 4 concurrent requests → aggregate generation throughput
3. **Progress display**: which workload is executing and rough ETA
4. **Suite parameters**: from settings defaults; deviations tag run as custom

## Files to create/modify

### New Files:
1. `monitor/long_context_workload.py` - long-context workload implementation
2. `monitor/burst_workload.py` - burst workload implementation
3. `monitor/settings.py` - settings storage and defaults for suite parameters

### Modify Files:
1. `monitor/benchmarks.py` - integrate long-context and burst workloads
2. `monitor/benchmarks/store.py` - extend to store suite parameters and tags
3. `monitor/web/routes.py` - add progress endpoint
4. `monitor/web/templates/dashboard.html` - add progress display
5. `monitor/settings.py` - store suite parameters in DB
6. `tests/test_benchmarks.py` - add tests for long-context and burst
7. `tests/fake_llama_server.py` - support concurrent requests and long prompts

## Design Decisions

1. **Suite parameters**: default 16k tokens for long-context, 4 concurrent requests for burst
2. **Progress**: track completed workloads, estimate based on previous workload times
3. **Tagging**: standard vs custom based on whether suite parameters match defaults
4. **Workload runner**: separate runner classes for each workload type, unified by interface

## Implementation Steps

1. Implement long-context workload (16k prompt, 32 tokens generation)
2. Implement burst workload (4 concurrent requests, measure aggregate throughput)
3. Add progress endpoint and display
4. Add suite settings storage and defaults
5. Wire settings into workload parameters
6. Implement tagging logic (standard vs custom)
7. Write tests for all three workloads including concurrency
8. Run make test && make smoke
