# Thor Monitor

A monitoring tool for this PC (nvidia-thor) that answers: which local LLM server (ollama or unsloth studio with llama.cpp) can serve which models, at what speed. It detects the running server and loaded model by probing — never by configuration — and never loads or changes models itself.

## Language

**Server**:
An LLM serving process this tool can detect and probe. Exactly two exist: `ollama serve` and `llama-server` (launched by unsloth studio).
_Avoid_: backend, engine, runtime

**Loaded Model**:
The model a Server currently holds in memory. At most one exists across all Servers at any moment; daemons may coexist idle.
_Avoid_: running model, active model

**Probe**:
A read-only check of processes, APIs, or system tools that establishes what is running and at what state.
_Avoid_: scan, poll

**Benchmark Run**:
A tool-initiated workload sent to the Loaded Model to measure speed. The only time the tool generates load. Runs on the default suite are *standard*; runs with modified parameters are tagged *custom* in the Catalog.
_Avoid_: test, evaluation

**Passive Observation**:
Continuous sampling of system stats and server telemetry without generating any load.
_Avoid_: monitoring (too broad — the whole tool monitors)

**Session**:
A period of real usage — user-driven traffic observed passively — recorded with its observed tok/s stats.
_Avoid_: conversation, chat

**Catalog**:
The accumulated history of Benchmark Runs and Sessions per model, stored in SQLite. Answers "which models ran, at what speed".
_Avoid_: report, log

**Unified Memory**:
System-wide RAM usage on the Jetson Thor, measured as `MemTotal − MemAvailable` from `/proc/meminfo`.
This is the same underlying kernel accounting that `tegrastats` uses for its RAM line,
but differs from `nvidia-smi`, which reports per-process GPU-side allocations.
_Avoid_: total memory, system memory
