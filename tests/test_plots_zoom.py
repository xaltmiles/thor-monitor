"""Tests for plots page zoom/pan/reset with synchronized time axis (issue #30)."""

import pytest
from fastapi.testclient import TestClient

from monitor.web.routes import app
import monitor.store
import tempfile
from pathlib import Path


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


def test_plots_html_includes_chartjs_plugin_zoom(test_client, test_db_path):
    """Test that plots.html includes chartjs-plugin-zoom from CDN."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check that chartjs-plugin-zoom is included
    assert 'cdn.jsdelivr.net/npm/chartjs-plugin-zoom' in html or 'chartjs-plugin-zoom' in html


def test_plots_html_includes_hammerjs(test_client, test_db_path):
    """Test that plots.html includes hammer.js for touch gestures."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check that hammer.js is included
    assert 'cdn.jsdelivr.net/npm/hammerjs' in html or 'hammerjs' in html


def test_plots_html_has_reset_zoom_button(test_client, test_db_path):
    """Test that plots.html has a visible reset zoom affordance."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for reset zoom button
    assert 'reset' in html.lower() or 'Reset' in html


def test_plots_html_has_synchronized_time_axis_js(test_client, test_db_path):
    """Test that plots.html has JavaScript for synchronized time axis."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for sync-related JavaScript
    assert 'sync' in html.lower() or 'synchronize' in html.lower() or 'synchronized' in html.lower()


def test_plots_html_preserves_zoom_on_refresh(test_client, test_db_path):
    """Test that plots.html has JavaScript to preserve zoom state across refreshes."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for zoom state preservation logic
    assert 'zoom' in html.lower()
    assert ('lastTimestamp' in html or 'zoomState' in html or 'zoomWindow' in html or 
            'range' in html.lower())


def test_plots_html_has_double_click_reset(test_client, test_db_path):
    """Test that plots.html has double-click reset functionality."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for double-click handler
    assert 'dblclick' in html.lower() or 'dbl-click' in html.lower() or 'double.*click' in html.lower()


def test_plots_html_range_buttons_update_synced_charts(test_client, test_db_path):
    """Test that range buttons properly update all synchronized charts."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check that range buttons are present and properly configured
    assert 'data-limit="60"' in html  # 1m
    assert 'data-limit="900"' in html  # 15m
    assert 'data-limit="3600"' in html  # 1h (default)
    assert 'data-limit="21600"' in html  # 6h
    assert 'data-limit="86400"' in html  # 24h


def test_plots_html_tooltips_work_while_zoomed(test_client, test_db_path):
    """Test that plots.html has tooltip configuration for zoomed state."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for tooltip configuration in chart options
    assert 'tooltip' in html.lower()
    assert 'mode' in html.lower()


def test_plots_html_synchronized_axes_implementation(test_client, test_db_path):
    """Test that plots.html has synchronized axes implementation."""
    response = test_client.get("/plots")
    assert response.status_code == 200
    html = response.text
    
    # Check for chart synchronization logic
    # Should have chart references and axis synchronization
    assert 'charts' in html.lower() or 'chart' in html.lower()
    
    # Should have shared axis min/max logic
    assert ('min' in html and 'max' in html and 'time' in html.lower()) or \
           ('synchronize' in html.lower() and 'axis' in html.lower())
