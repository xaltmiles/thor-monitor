#!/usr/bin/env bash
# Smoke test: run the real app against the real system and assert it works
# end to end. Catches the bug class fixture-fed unit tests cannot: unwired
# sources, broken package imports, template errors, dead real-system paths.
#
# Usage: bash scripts/smoke.sh   (or: make smoke)
set -euo pipefail

PORT=8000
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp /tmp/monitor-smoke.XXXXXX.log)"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# Abort on an occupied port rather than killing an unknown process.
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3>&- 3<&- || true
  fail "port ${PORT} already in use; stop the running monitor first"
fi

cleanup() {
  fuser -k ${PORT}/tcp >/dev/null 2>&1 || true
  sleep 1
  fuser -k -9 ${PORT}/tcp >/dev/null 2>&1 || true
  rm -f "$LOG"
}
trap cleanup EXIT

# --- 1. The app starts and serves the dashboard -------------------------
uv run monitor >"$LOG" 2>&1 &
code="000"
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$BASE/" 2>/dev/null || echo "000")
  [ "$code" = "200" ] && break
  kill -0 "$(jobs -p)" 2>/dev/null || { cat "$LOG" >&2; fail "monitor exited during startup"; }
  sleep 1
done
[ "$code" = "200" ] || { tail -20 "$LOG" >&2; fail "dashboard never returned 200 (last code: $code)"; }

# --- 2. Dashboard renders real content (template compiles + executes) ----
curl -s --max-time 5 "$BASE/" | grep -q "Monitor Dashboard" \
  || fail "dashboard HTML missing expected content (template broken?)"

# --- 3. Telemetry flows: sampler -> store -> rows appear -----------------
dbcheck() {
  uv run python - <<'PY' 2>/dev/null || echo "DBFAIL"
import sqlite3
from monitor.store import DB_PATH
db = sqlite3.connect(str(DB_PATH))
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
missing = {"telemetry_samples", "sessions", "models", "settings"} - tables
if missing:
    print(f"DBFAIL missing tables: {sorted(missing)}")
else:
    print(db.execute("SELECT COUNT(*) FROM telemetry_samples").fetchone()[0])
PY
}
before=$(dbcheck)
[ "$before" = "DBFAIL" ] && fail "store schema check failed (missing tables?)"
case "$before" in (*DBFAIL*) fail "store schema check failed";; esac
[ "$before" -ge 0 ] 2>/dev/null || fail "store row count unreadable: $before"

sleep 4
after=$(dbcheck)
case "$after" in (*DBFAIL*) fail "store schema check failed after sampling";; esac
[ "$after" -gt "$before" ] || fail "telemetry_samples did not grow during smoke ($before -> $after); sampler or store broken"

# --- 4. Latest-sample payload carries core fields ------------------------
curl -s --max-time 5 "$BASE/api/telemetry/latest" | grep -q '"memory_total"' \
  || fail "/api/telemetry/latest missing memory_total"

# --- 5. Probes run crash-free against whatever is on this machine --------
curl -s --max-time 20 "$BASE/api/probes/detect" | grep -q '"servers"' \
  || fail "/api/probes/detect missing servers key"

# --- 6. Runtime state lives in the store ---------------------------------
[ -f "$HOME/.monitor/monitor.db" ] || fail "store missing at ~/.monitor/monitor.db"

echo "SMOKE PASS: dashboard 200, telemetry flowing ($before -> $after rows), probes OK, store in place"
