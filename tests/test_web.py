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
async def test_dashboard_html_renders_ollama_stats(test_client):
    """Test that dashboard renders ollama stats when present and ollama is detected."""
    import json
    from monitor.store import insert_telemetry_sample
    
    # Insert a telemetry sample with ollama stats
    ollama_stats = {
        "ollama_prompt_tokens": 1000,
        "ollama_generated_tokens": 500,
        "ollama_prompt_tokens_rate": 10.0,
        "ollama_generated_tokens_rate": 5.0,
        "ollama_prompt_tokens_rate_avg": 10.5,
        "ollama_generated_tokens_rate_avg": 5.2
    }
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory="[]",
        ollama_stats=json.dumps(ollama_stats)
    )
    
    response = test_client.get("/")
    
    assert response.status_code == 200
    html = response.text
    
    # When ollama stats are present and ollama is detected, the template should display ollama rates
    # Assert the actual rate values (10.5 and 5.2) are rendered
    assert "10.5" in html, "Expected ollama_prompt_tokens_rate_avg=10.5 to be rendered"
    assert "5.2" in html, "Expected ollama_generated_tokens_rate_avg=5.2 to be rendered"


@pytest.mark.asyncio
async def test_dashboard_html_renders_llama_stats(test_client):
    """Test that dashboard renders llama-server stats when present."""
    import json
    from monitor.store import insert_telemetry_sample
    
    # Insert a telemetry sample with llama stats
    llama_stats = {
        "prompt_tokens": 10000,
        "generated_tokens": 5000,
        "speculative_accepts": 500,
        "prompt_tokens_rate": 10.0,
        "generated_tokens_rate": 5.0,
        "speculative_accepts_rate": 0.5,
        "prompt_tokens_rate_avg": 10.5,
        "generated_tokens_rate_avg": 5.2,
        "speculative_accepts_rate_avg": 0.6  # rounded to 1 decimal place
    }
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory="[]",
        llama_stats=json.dumps(llama_stats)
    )
    
    response = test_client.get("/")
    
    assert response.status_code == 200
    html = response.text
    
    # When llama stats are present, the template should display llama-server rates
    # Assert the actual rate values (10.5, 5.2, 0.6) are rendered (0.6 = round(0.6, 1))
    assert "10.5" in html, "Expected prompt_tokens_rate_avg=10.5 to be rendered"
    assert "5.2" in html, "Expected generated_tokens_rate_avg=5.2 to be rendered"
    assert "0.6" in html, "Expected speculative_accepts_rate_avg=0.6 to be rendered"


@pytest.mark.asyncio
async def test_api_telemetry_latest_empty(test_client):
    """Test API returns empty when no data exists."""
    response = test_client.get("/api/telemetry/latest")
    
    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
async def test_no_htmx_json_dump_pattern_in_template():
    """Test that dashboard template has no hx-swap='outerHTML' on telemetry endpoints."""
    import re
    from pathlib import Path
    
    # Read the dashboard template
    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()
    
    # Check that no hx-swap='outerHTML' appears on any hx-get for /api/telemetry endpoints
    # Pattern: hx-get="api/telemetry/..." ... hx-swap="outerHTML"
    pattern = r'hx-get=["\']/(?:api/telemetry/[^"\']*)["\'][^>]*hx-swap=["\']outerHTML["\']'
    matches = re.findall(pattern, content)
    
    assert matches == [], (
        f"Found {len(matches)} problematic htmx patterns in dashboard.html. "
        "hx-swap='outerHTML' should not be used on /api/telemetry endpoints "
        "as they return raw JSON that would be rendered as text. "
        f"Matches: {matches}"
    )


@pytest.mark.parametrize(
    "url,expected_active",
    [
        ("/", "dashboard"),
        ("/catalog/comparison", "comparison"),
        ("/catalog/timeline", "timeline"),
        ("/settings", "settings"),
        ("/plots", "plots"),
    ],
)
def test_nav_active_class_per_page(test_client, url, expected_active):
    """Test that each page highlights the correct navigation link as active."""
    response = test_client.get(url)
    assert response.status_code == 200
    html = response.text
    
    # Assert the combined rendered markup: <a href="{url}" class="active">
    # This ensures the *specific* link for this page is marked active
    assert f'<a href="{url}" class="active">' in html


@pytest.mark.asyncio
async def test_plots_html_returns_200_with_charts(test_client, test_db_path):
    """Test that /plots returns 200 and contains expected chart containers."""
    # Insert test telemetry data
    from monitor.store import insert_telemetry_sample
    import json
    
    # Insert sample telemetry with both llama and ollama stats
    for i in range(10):
        llama_stats = {
            "prompt_tokens": 1000 + i * 100,
            "generated_tokens": 500 + i * 50,
            "speculative_accepts": 50 + i * 5,
            "prompt_tokens_rate": 10.0 + i * 0.5,
            "generated_tokens_rate": 5.0 + i * 0.2,
            "speculative_accepts_rate": 0.5 + i * 0.05,
        }
        ollama_stats = {
            "ollama_prompt_tokens": 800 + i * 80,
            "ollama_generated_tokens": 400 + i * 40,
            "ollama_prompt_tokens_rate": 8.0 + i * 0.4,
            "ollama_generated_tokens_rate": 4.0 + i * 0.2,
        }
        
        await insert_telemetry_sample(
            memory_total=32_000_000_000,
            memory_free=16_000_000_000 - i * 500_000_000,
            memory_used=16_000_000_000 + i * 500_000_000,
            gpu_util=45.0 + i * 2,
            gpu_temp=70.0 + i,
            gpu_power=150.0 + i * 5,
            process_memory="[]",
            llama_stats=json.dumps(llama_stats) if i % 2 == 0 else None,
            ollama_stats=json.dumps(ollama_stats) if i % 2 == 1 else None,
        )
    
    response = test_client.get("/plots")
    
    assert response.status_code == 200
    html = response.text
    
    # Check for expected chart containers
    assert 'id="memory-chart"' in html
    assert 'id="gpu-temp-chart"' in html
    assert 'id="llama-tok-s-chart"' in html
    assert 'id="ollama-tok-s-chart"' in html
    
    # Check that Chart.js is included
    assert 'cdn.jsdelivr.net/npm/chart.js' in html
    
    # Assert the combined rendered markup: <a href="/plots" class="active">
    assert '<a href="/plots" class="active">' in html
    
    # Check for range selector buttons
    assert 'data-limit="60"' in html  # 1m
    assert 'data-limit="300"' in html  # 5m
    assert 'data-limit="900"' in html  # 15m
    assert 'data-limit="3600"' in html  # 1h
