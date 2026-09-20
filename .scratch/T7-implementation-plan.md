# T7 Implementation Plan: Ollama Parity

## Acceptance Criteria
- [ ] Benchmark suite runs against ollama when it is the detected Server, results stored identically
- [ ] Passive tok/s parsed from ollama debug logs; Sessions recorded for ollama usage
- [ ] Guidance note shown when ollama runs without debug logging
- [ ] One-Loaded-Model warning works across mixed Server states (ollama loaded + llama-server loaded)
- [ ] Tests use the fake LLM server's ollama-mode endpoints and fixture log streams

## Current State
- Benchmark runners (workloads.py) only support llama-server's OpenAI-compatible endpoint
- Passive tok/s only works for llama-server via Prometheus /metrics
- Sessions only recorded for llama-server
- No ollama-specific handling in benchmark runner

## What Needs to Be Added

### 1. Ollama Benchmark Support
- Create ollama-specific workload runners (workloads.py)
  - Use ollama's native `/api/generate` endpoint
  - Support short, long-context, burst workloads
- Update BenchmarkRunner to detect server type and route to correct runner
- Update benchmarks.py to handle ollama server type in _create_run

### 2. Ollama Passive Tok/s
- Parse ollama server logs for tok/s metrics
- Ollama debug logging format needs to be parsed
- Create OllamaStatsSource similar to LLaMAStatsSource

### 3. Guidance for Debug Logging
- When ollama detected without debug logging, show guidance note
- Need to detect if ollama has debug logging enabled
- Add warning in probe result or separate guidance endpoint

### 4. Mixed Server Warning
- One-Loaded-Model warning already exists (ProbeResult.warning)
- Should work across mixed server types (ollama + llama-server)
- Need to verify this works correctly

## Files to Modify

### monitor/workloads.py
- Add OllamaShortWorkloadRunner (ollama /api/generate)
- Add OllamaLongContextWorkloadRunner
- Add OllamaBurstWorkloadRunner
- Add count_tokens_in_ollama_sse_chunk helper

### monitor/benchmarks.py
- Update _create_run to handle ollama server type
- Update _run_benchmark to detect server type and use appropriate runner
- Update _make_short_runner to return correct runner based on server type

### monitor/sessions.py
- Add ollama passive tok/s parsing
- Parse ollama debug log format for token stats

### monitor/probes/
- Add ollama debug logging detection
- Add guidance note generation

### tests/
- Add FakeOllamaServer (similar to FakeLLaMAServer)
- Add test_ollama_benchmarks.py

## Implementation Order
1. FakeOllamaServer (for testing)
2. Ollama workload runners in workloads.py
3. BenchmarkRunner updates in benchmarks.py
4. Sessions.py ollama support
5. Probe guidance for debug logging
6. Tests
7. Run full test suite
8. Commit
