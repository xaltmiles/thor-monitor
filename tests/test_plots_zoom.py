"""Behavioral tests for the plots page zoom/pan/reset (issue #30).

These tests execute the template's real inline script under the same library
versions the template pins on the CDN (Chart.js 4.5.1, chartjs-plugin-zoom
2.0.1, hammerjs 2.0.8, luxon 3.7.2), inside jsdom, and drive it with real DOM
events: wheel zoom, shift+drag pan, drag zoom, reset button/dblclick, and
simulated 30s refreshes with advancing telemetry.

Substring greps over the served HTML cannot catch what previous review rounds
found here (a reset that silently froze live tracking; wheel zoom moving the
value axis), because the bugs live in the executed script, not the markup.
The harness in tests/js/run_plots_tests.mjs runs the actual code path; each
scenario below is one scenario of that harness and fails if the page's real
JavaScript misbehaves under the real Chart.js.

Setup: node must be on PATH and `npm install` run at the repo root
(package.json pins the exact library versions). Without them these tests
skip; with them they are mandatory pass/fail.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from monitor.web.routes import app
import monitor.store
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tests" / "js" / "run_plots_tests.mjs"

# One pytest test per harness scenario: each scenario boots a fresh jsdom page
# and drives one acceptance criterion (or regression) end to end.
SCENARIOS = [
    # Sanity: the inline script builds six real Chart instances on a time axis
    "charts_build",
    # AC1+AC2: wheel zoom narrows one chart and all six follow; y axis fixed
    "wheel_zoom_syncs_all",
    # AC1+AC2: shift+drag pan propagates the window to every chart
    "shift_drag_pan_syncs_all",
    # AC1+AC2: plain-drag zoom syncs all charts to the selected window
    "drag_zoom_syncs_all",
    # AC3: Reset Zoom button and double-click reset all charts and clear state
    "reset_button_and_dblclick",
    # AC4: zoom survives incremental refresh and is clamped on range switch
    "zoom_survives_refresh_and_range_switch",
    # Regression: after a reset the charts keep tracking new telemetry
    # (reset used to resurrect the zoom window and freeze the dashboard)
    "reset_does_not_freeze_tracking",
    # AC5: tooltips stay usable while zoomed (real mousemove activates them)
    "tooltip_works_while_zoomed",
]


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_db = temp_dir / "test_monitor.db"
    original_path = monitor.store.DB_PATH

    monitor.store.DB_PATH = temp_db

    yield temp_db

    import shutil
    shutil.rmtree(temp_dir)
    monitor.store.DB_PATH = original_path


@pytest.fixture
def test_client(test_db_path):
    """Create a test client with test database."""
    import asyncio
    from monitor.store import init_db

    # Initialize test database
    asyncio.run(init_db())

    return TestClient(app)


def _run_scenario(test_client, scenario: str) -> None:
    """Serve /plots and execute one harness scenario against its real JS."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node executable not on PATH; required for plots behavioral tests")
    if not (REPO_ROOT / "node_modules" / "jsdom").is_dir():
        pytest.skip("node_modules missing; run `npm install` at the repo root (see package.json)")

    response = test_client.get("/plots")
    assert response.status_code == 200

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(response.text)
        html_path = f.name

    try:
        proc = subprocess.run(
            [node, str(HARNESS), html_path, scenario],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
        )
    finally:
        Path(html_path).unlink(missing_ok=True)

    detail = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"harness scenario '{scenario}' failed:\n{detail}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_plots_page_behavior(test_client, test_db_path, scenario):
    """Run one behavioral scenario of the plots-page JS harness."""
    _run_scenario(test_client, scenario)


def test_plots_cdn_versions_match_test_dependencies(test_client, test_db_path):
    """The jsdom harness tests the versions the template actually pins.

    If the template's CDN pins drift from package.json, the behavioral tests
    would silently exercise the wrong library versions.
    """
    html = test_client.get("/plots").text
    pkg = json.loads((REPO_ROOT / "package.json").read_text())

    for name, version in pkg["devDependencies"].items():
        if name == "canvas" or name == "jsdom":
            continue  # test-infrastructure deps, not loaded by the template
        expected = f"cdn.jsdelivr.net/npm/{name}@{version}"
        assert expected in html, (
            f"template does not pin {name}@{version} from the CDN; "
            f"the behavioral harness (package.json) would test a different version"
        )
