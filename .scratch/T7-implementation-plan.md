# T7 Implementation Plan

## Changes Made

### 1. RealOllamaStatsSource namespacing (BLOCKER)
- Fix: namespaced keys to `ollama_prompt_tokens`, `ollama_generated_tokens_rate`
- Files: monitor/telemetry/real.py:~315

### 2. FixtureOllamaStatsSource namespacing
- Already namespaced; needs no change

### 3. Sampler routing
- Files: monitor/sampler.py:64-65
- Already checks `ollama_prompt_tokens` for detection

### 4. Remove scratch files
- Delete: T7.md, T7_FIXES.md, T7_COMPLETE.md, T7_COMPLETE_REPORT.md, T7_FINAL.md, T7_IMPLEMENTATION.md, T7_STATUS.md, T7_SUBMISSION.md, T7_SUMMARY.md

### 5. Add tests
- Test RealOllamaStatsSource keys
- Test sampler ollama routing with real source

### 6. Fix unbounded file reading
- Limit to last 10000 lines in RealOllamaStatsSource._read_log_file
