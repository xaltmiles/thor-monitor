# Fix Ticket #7 Issues

## Issues to Fix

1. **BLOCKER**: Progress display shows no active workload/ETA during run - workload_results only written at END
2. **MAJOR**: Custom tagging not enforced - no comparison to defaults
3. **MAJOR**: New workload tests don't actually test workloads (fake server port 18080, no listener)
4. **MAJOR**: Long-context prompt tok/s likely ~0 (measures chunks before TTFT, but llama-server streams nothing before first token)
5. **MINOR**: Dead duplicated `if not results:` in BurstWorkloadRunner
6. **MINOR**: _count_tokens_in_chunk duplicated across runners
7. **MINOR**: server_port constructor params are dead (run() overwrites)
8. **MINOR**: workload_results SQL line mangled
9. **MINOR**: Failed workloads return zero-filled results silently (no abort)

## Implementation Plan

### 1. Fix Progress Display (BLOCKER)
- Store partial workload results during run progress
- Use state machine to track completed workloads
- Update stored row incrementally

### 2. Enforce Custom Tagging (MAJOR)
- Compare actual parameters to defaults
- Set tag="custom" if any deviation
- Ensure standard_run=False when custom

### 3. Fix Workload Tests (MAJOR)
- Use correct fake server port in tests
- Assert actual metrics > 0
- Extend e2e test to verify all three workloads
- Add progress endpoint tests

### 4. Fix Long-Context Prompt Tok/s (MAJOR)
- Track total time elapsed before first token
- Use usage.prompt_tokens from first chunk
- Calculate prompt_tok_s = prompt_tokens / time_before_first_token

### 5-9. Minor cleanup and improvements
