# Detect servers and models by probing — never by declaring them in configuration

The tool discovers which LLM server is running (ollama or unsloth's llama-server) and which model is loaded by probing processes, APIs, and system tools. The dashboard's configuration page governs tool settings only — benchmark suite parameters, sampling rate, warning thresholds — never what server or model is running. Declaring the server/model in config would be far simpler, but this box is a test bench where the operator swaps servers and models constantly; a declared config rots with every swap, and half the value of the tool is that it always reflects reality without being told.

## Considered Options

- **Configuration page with server+model declaration (rejected)**: goes stale the moment the operator switches server or model, and silently lies when it does.
- **Hybrid — declared config with probe verification (rejected)**: two sources of truth; when they disagree, which wins?

## Consequences

- Every new server type requires a new probe (process signature + model-identification path); ollama and llama.cpp are the only two in scope.
- The at-most-one-Loaded-Model invariant is assumed and warned upon, not enforced.
- Benchmark Run parameters are configurable via the configuration page (persisted in SQLite — there is deliberately no config file). Runs deviating from the default suite are tagged *custom* in the Catalog, keeping standard-suite cross-model comparisons clean.
