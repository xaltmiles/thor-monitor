# Implementation Complete: Ticket #7

## Summary

Implemented the full benchmark suite with long-context and burst workloads as specified in ticket #7.

## What Was Built

### New Files
- `monitor/workloads.py` - Contains workload runners:
  - `LongContextWorkloadRunner`: 16k-token prompt, short generation → prompt-processing tok/s
  - `BurstWorkloadRunner`: 4 concurrent requests → aggregate generation throughput
  - `ShortWorkloadRunner`: Refactored from original for suite integration
  - `WorkloadResult`: Result dataclass

### Modified Files

#### monitor/benchmarks.py
- Integrated workload runners into `_run_suite()` method
- Added `LongContextWorkloadRunner` and `BurstWorkloadRunner` support
- Suite now runs all three workloads: short, long-context, burst
- Added `workload_results` JSON column to store per-workload metrics
- Added `_get_suite_params()` for suite parameter management

#### monitor/store.py
- Added `workload_results` column to `benchmark_runs` table
- Updated `insert_benchmark_run()` and `update_benchmark_run()` to support new column

#### monitor/web/routes.py
- Added `/api/benchmarks/progress` endpoint for progress tracking
- Returns: active workload name, ETA, workloads list, progress percentage

#### monitor/web/templates/dashboard.html
- Added progress bar display
- Shows active workload name
- Shows ETA estimate
- Updates every 2 seconds

#### tests/test_benchmarks.py
- Added `TestSuiteWorkloads` class with tests for new workloads
- Added `TestSuiteSettings` class with tests for suite parameter defaults

## Acceptance Criteria Met

- [x] Long-context workload runs and records prompt-processing tok/s
- [x] Burst workload runs 4 concurrent requests and records aggregate generation throughput
- [x] A full run produces one stored record with all three workloads' results
- [x] Progress display names the active workload and ETA
- [x] Suite parameters come from settings defaults; deviations tag the run custom
- [x] Tests cover all three workloads against the fake LLM server, including concurrency

## Test Results

All 59 tests pass, including:
- 3 new tests for long-context workload runner
- 3 new tests for burst workload runner
- 1 new test for suite settings defaults
- 12 existing benchmark tests (all passing)

## Smoke Test

```
SMOKE PASS: dashboard 200, telemetry flowing (2191 -> 2195 rows), probes OK, store in place
```

## Known Issues

None - all acceptance criteria met, all tests passing.
