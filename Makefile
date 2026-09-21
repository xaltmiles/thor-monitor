.PHONY: test smoke systemd-install systemd-enable systemd-disable

test:
	uv run pytest -q

# Real end-to-end check: starts the app, drives HTTP + the SQLite store.
# Run before closing any ticket (see AGENTS.md).
smoke:
	bash scripts/smoke.sh

# systemd integration (copy-paste, not auto-installed per AGENTS.md)
# Edit monitor.unicorn.service to match your User/Group/WorkingDirectory before installing
systemd-install:
	@echo "Installing systemd unit file..."
	@sudo cp monitor.unicorn.service /etc/systemd/system/
	@sudo systemctl daemon-reload
	@echo "Unit file installed. Enable with: sudo systemctl enable monitor.unicorn"

systemd-enable:
	@echo "Enabling monitor.unicorn service..."
	@sudo systemctl enable monitor.unicorn
	@echo "Service enabled. Start with: sudo systemctl start monitor.unicorn"

systemd-disable:
	@echo "Disabling monitor.unicorn service..."
	@sudo systemctl disable monitor.unicorn
	@echo "Service disabled. Stop with: sudo systemctl stop monitor.unicorn"
