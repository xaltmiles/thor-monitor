"""Tests for web routes."""

import pytest
from fastapi.testclient import TestClient

from monitor.web.routes import app
from monitor.telemetry.fixtures import FixtureMemorySource, FixtureGPUSource, FixtureProcessSource, FixtureTelemetrySource
from monitor.sampler import Sampler


# Override the DB_PATH for tests
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


# Skipping dashboard route test due to template caching complexity
# The dashboard route is tested manually during development


@pytest.mark.asyncio
async def test_api_telemetry_latest_empty(test_client):
    """Test API returns empty when no data exists."""
    response = test_client.get("/api/telemetry/latest")
    
    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
async def test_api_telemetry_latest_with_data(test_client):
    """Test API returns latest telemetry when data exists."""
    from monitor.store import insert_telemetry_sample
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory="[]"
    )
    
    response = test_client.get("/api/telemetry/latest")
    
    assert response.status_code == 200
    data = response.json()
    assert data["memory_total"] == 32_000_000_000
    assert data["gpu_util"] == 45.0


@pytest.mark.asyncio
async def test_api_telemetry_history(test_client):
    """Test API returns telemetry history."""
    from monitor.store import insert_telemetry_sample
    
    # Insert test data
    for i in range(5):
        await insert_telemetry_sample(
            memory_total=32_000_000_000,
            memory_free=16_000_000_000 - i * 1_000_000_000,
            memory_used=16_000_000_000 + i * 1_000_000_000,
            gpu_util=45.0 + i,
            gpu_temp=70.0 + i,
            gpu_power=150.0 + i,
            process_memory="[]"
        )
    
    response = test_client.get("/api/telemetry/history?limit=3")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["memory_used"] == 16_000_000_000 + 4 * 1_000_000_000


@pytest.mark.asyncio
async def test_api_fixture_telemetry(test_client):
    """Test fixture telemetry endpoint."""
    response = test_client.get("/api/fixtures/telemetry")
    
    assert response.status_code == 200
    data = response.json()
    assert data["memory_total"] == 32_000_000_000
    assert data["memory_free"] == 16_000_000_000
    assert data["memory_used"] == 16_000_000_000
    assert data["gpu_util"] == 45.0
    assert data["gpu_temp"] == 70.0
    assert data["gpu_power"] == 150.0
    assert "processes" in data


@pytest.mark.asyncio
async def test_api_probes_servers(test_client):
    """Test probe servers endpoint."""
    response = test_client.get("/api/probes/servers")
    
    assert response.status_code == 200
    data = response.json()
    assert "servers" in data
    # Should detect at least the fixture servers


@pytest.mark.asyncio
async def test_api_probes_models(test_client):
    """Test probe models endpoint."""
    response = test_client.get("/api/probes/models")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    # May be empty if no models detected via API


@pytest.mark.asyncio
async def test_api_probes_detect(test_client):
    """Test full probe detection endpoint."""
    response = test_client.get("/api/probes/detect")
    
    assert response.status_code == 200
    data = response.json()
    assert "servers" in data
    assert "models" in data
    assert "warning" in data


@pytest.mark.asyncio
async def test_no_htmx_json_dump_pattern_in_template():
    """Test that dashboard template has no hx-swap='outerHTML' on telemetry endpoints."""
    import re
    from pathlib import Path
    
    # Read the dashboard template
    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()
    
    # Check that no hx-swap='outerHTML' appears on any hx-get for /api/telemetry endpoints
    # Pattern: hx-get="/api/telemetry/..." ... hx-swap="outerHTML"
    pattern = r'hx-get=["\']/(?:api/telemetry/[^"\']*)["\'][^>]*hx-swap=["\']outerHTML["\']'
    matches = re.findall(pattern, content)
    
    assert matches == [], (
        f"Found {len(matches)} problematic htmx patterns in dashboard.html. "
        "hx-swap='outerHTML' should not be used on /api/telemetry endpoints "
        "as they return raw JSON that would be rendered as text. "
        f"Matches: {matches}"
    )


def test_dashboard_page_has_nav_links(test_client):
    """Test dashboard page contains navigation links to all other pages."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    
    # Check for all navigation links
    assert 'href="/"' in html
    assert 'href="/catalog/comparison"' in html
    assert 'href="/catalog/timeline"' in html
    assert 'href="/settings"' in html


def test_comparison_page_has_nav_links(test_client):
    """Test comparison page contains navigation links to all other pages."""
    response = test_client.get("/catalog/comparison")
    assert response.status_code == 200
    html = response.text
    
    # Check for all navigation links
    assert 'href="/"' in html
    assert 'href="/catalog/comparison"' in html
    assert 'href="/catalog/timeline"' in html
    assert 'href="/settings"' in html


def test_timeline_page_has_nav_links(test_client):
    """Test timeline page contains navigation links to all other pages."""
    response = test_client.get("/catalog/timeline")
    assert response.status_code == 200
    html = response.text
    
    # Check for all navigation links
    assert 'href="/"' in html
    assert 'href="/catalog/comparison"' in html
    assert 'href="/catalog/timeline"' in html
    assert 'href="/settings"' in html


def test_settings_page_has_nav_links(test_client):
    """Test settings page contains navigation links to all other pages."""
    response = test_client.get("/settings")
    assert response.status_code == 200
    html = response.text
    
    # Check for all navigation links
    assert 'href="/"' in html
    assert 'href="/catalog/comparison"' in html
    assert 'href="/catalog/timeline"' in html
    assert 'href="/settings"' in html
