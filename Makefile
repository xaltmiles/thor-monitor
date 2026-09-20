.PHONY: test smoke

test:
	uv run pytest -q

# Real end-to-end check: starts the app, drives HTTP + the SQLite store.
# Run before closing any ticket (see AGENTS.md).
smoke:
	bash scripts/smoke.sh
