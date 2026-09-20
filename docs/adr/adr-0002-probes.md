# ADR-0002: Probe-based Server and Model Detection

## Status

Accepted

## Context

Issue #3 requires the tool to detect which Server is running (`ollama serve`, `llama-server`) and which model it has Loaded, purely by probing processes and APIs—never by configuration. This extends the walking skeleton from issue #2.

## Decision

We implement a **Probe** system:

- **ServerDetector**: Scans processes to identify running servers (ollama, llama-server)
- **ModelDetector**: Resolves Loaded Model identity
  - ollama: Query `/api/running-models` endpoint
  - llama-server: Extract model path from cmdline + query model info
- **ProbeResult**: Unified output with server type, model info, and detection status
- **Multi-model warning**: When >1 Loaded Model detected across servers

### Module Structure

```
monitor/
├── probes/
│   ├── __init__.py
│   ├── interface.py      # Abstract probe interfaces
│   ├── server.py         # Server detection via process scan
│   ├── model.py          # Model identity resolution
│   └── fixtures.py       # Test fixtures (fake server APIs, process tables)
├── web/
│   └── routes.py         # /api/probes endpoints
└── probes.py             # High-level probe orchestration
```

### Data Flow

1. **Server detection** → Process table scan → identify ollama/llama-server
2. **Model detection** → Server-specific probe → resolve model identity
3. **Store** → Insert/update models in DB
4. **Dashboard** → Display Loaded Model card + multi-model warning

### Key Decisions

- **No config files**: All detection by probing only (per ADR-0001)
- **Stateless probes**: Each probe call is independent; no cached state
- **Recovery on restart/swap**: Probes re-run on demand; no restart of tool required
- **Per-server probes**: Each server type has its own detection logic

### API Endpoints

- `GET /api/probes/servers` → List detected servers
- `GET /api/probes/models` → Get Loaded Model info
- `GET /api/probes/detect` → Full detection run

## Consequences

### Positive

- Detection is always current; no stale config
- Survives server restarts and model swaps
- User never needs to declare what's running

### Negative

- Requires ollama debug logging for passive tok/s (operator action)
- llama-server must expose Prometheus metrics

## Testing

- Fixture process tables for deterministic server detection
- Fake HTTP servers for ollama/llama-server APIs
- Tests verify recovery after server restart and model swap
