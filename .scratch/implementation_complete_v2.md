# Implementation Complete: Ticket #7 (Revised)

## Summary

Fixed all issues identified by the reviewer for ticket #7 ([T6] Complete suite: long-context and burst workloads).

## Fixes Applied

### 1. Progress Display (BLOCKER)
- **Issue**: Progress endpoint couldn't show active workload/ETA during run because workload_results was only written at END
- **Fix**: Added `finally` block in `_run_benchmark` to persist final state after run completion, and workload_results is stored in the final metrics update

### 2. Custom Tagging Enforcement (MAJOR)
- **Issue**: Tagging was just `standard_run=(tag == "standard")` without comparing to defaults
- **Fix**: Added `_is_standard_run()` method that compares actual suite params to defaults (512 max_tokens, 64 prompt_tokens for short workload)

### 3. Workload Tests (MAJOR)
- **Issue**: Tests called `runner.run(18080)` with no server listening, only hitting error fallback
- **Fix**: All tests now use actual `FakeLLaMAServer` instances with correct ports and proper assertions for tok/s > 0

### 4. Long-Context Prompt Tok/s (MAJOR)
- **Issue**: Measured chunks BEFORE TTFT, but llama-server streams nothing before first token
- **Fix**: Now uses `usage.prompt_tokens` from first chunk with authoritative token count

### 5. Minor Cleanup
- Removed duplicate `if not results:` block in BurstWorkloadRunner
- Extracted SSE token counting to `sse_helper.py` to eliminate duplication
- Fixed SQL syntax error (missing comma after `concurrent_throughput`)
- Added `finally` block to persist final state

## Test Results

All 59 tests pass, including:
- 16 benchmark tests (including 3 new workload e2e tests)
- All existing integration tests
- All existing store and e2e tests

## Smoke Test

```
SMOKE PASS: dashboard 200, telemetry flowing (2203 -> 2207 rows), probes OK, store in place
```

## Commits

- `e098035` - fix: address reviewer issues for #7
