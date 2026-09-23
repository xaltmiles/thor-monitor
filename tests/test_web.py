"""Tests for web routes."""

import pytest
from fastapi.testclient import TestClient

from monitor.web.routes import app
from monitor.telemetry.fixtures import FixtureMemorySource, FixtureGPUSource, FixtureProcessSource, FixtureTelemetrySource
from monitor.sampler import Sampler
from monitor.store import insert_telemetry_sample


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
    
    # Check for range selector buttons (1m, 15m, 1h, 6h, 24h - with 1h as default)
    assert 'data-limit="60"' in html  # 1m
    assert 'data-limit="900"' in html  # 15m
    assert 'data-limit="3600"' in html  # 1h (default)
    assert 'data-limit="21600"' in html  # 6h
    assert 'data-limit="86400"' in html  # 24h
    assert 'data-default="true"' in html  # 1h is the default


def test_dashboard_template_has_separate_last_rate_elements():
    """Test that dashboard template has separate last rate elements for prompt and generation."""
    import re
    from pathlib import Path

    # Read the dashboard template
    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()

    # Check for separate last rate elements for llama.cpp
    assert 'id="tok-last-prompt-llama"' in content, "Expected tok-last-prompt-llama element"
    assert 'id="tok-last-gen-llama"' in content, "Expected tok-last-gen-llama element"

    # Check for separate last rate elements for ollama
    assert 'id="tok-last-prompt-ollama"' in content, "Expected tok-last-prompt-ollama element"
    assert 'id="tok-last-gen-ollama"' in content, "Expected tok-last-gen-ollama element"

    # Ensure old single shared element is removed
    assert 'id="tok-last-rate-llama"' not in content, "Old shared tok-last-rate-llama should be removed"
    assert 'id="tok-last-rate-ollama"' not in content, "Old shared tok-last-rate-ollama should be removed"


def test_dashboard_template_last_rate_style():
    """Test that last rate elements have proper styling for independent display."""
    from pathlib import Path

    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()

    # Check that last rate elements have margin-top for spacing
    assert 'margin-top: 10px' in content, "Expected margin-top for last rate elements"


def test_dashboard_template_prompt_and_gen_metrics_structure():
    """Test that prompt and generation metrics have correct structure."""
    from pathlib import Path

    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()

    # Check llama.cpp metrics structure
    assert 'id="tok-prompt-llama"' in content
    assert 'id="tok-generate-llama"' in content

    # Check ollama metrics structure
    assert 'id="tok-prompt-ollama"' in content
    assert 'id="tok-generate-ollama"' in content


def test_dashboard_template_has_update_last_rate_display_function():
    """Test that dashboard template has the updateLastRateDisplay function."""
    from pathlib import Path

    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()

    # Check for the updateLastRateDisplay function
    assert 'function updateLastRateDisplay' in content, "Expected updateLastRateDisplay function"

    # Check for the lastNonZeroRates tracking structure
    assert 'lastNonZeroRates' in content, "Expected lastNonZeroRates tracking"
    assert 'lastNonZeroTimes' in content, "Expected lastNonZeroTimes tracking"

    # Check for separate tracking for llama and ollama
    assert 'llama: { prompt' in content, "Expected separate llama prompt tracking"
    assert 'ollama: { prompt' in content, "Expected separate ollama prompt tracking"


def test_dashboard_template_has_last_rate_interval():
    """Test that dashboard template has interval for updating last rate display."""
    from pathlib import Path

    template_path = Path(__file__).parent.parent / "monitor" / "web" / "templates" / "dashboard.html"
    content = template_path.read_text()

    # Check for setInterval for last rate updates
    assert 'setInterval' in content, "Expected setInterval for last rate updates"
    assert 'updateLastRateDisplay' in content, "Expected updateLastRateDisplay call in interval"

    # Check that last rate display updates for both servers and metrics
    assert "'llama', 'ollama'" in content or "['llama', 'ollama']" in content, "Expected both servers in interval"
    assert "'prompt', 'generate'" in content or "['prompt', 'generate']" in content, "Expected both metrics in interval"


@pytest.mark.asyncio
async def test_dashboard_last_rate_persists_and_ticking(test_client, test_db_path):
    """Test that Last rate lines persist once non-zero rate is seen and Ys ago ticks."""
    from monitor.store import insert_telemetry_sample
    import json
    
    # Insert telemetry with non-zero rates to trigger the "Last" lines
    llama_stats = {
        "prompt_tokens": 1000,
        "generated_tokens": 500,
        "prompt_tokens_rate": 10.5,
        "generated_tokens_rate": 5.2,
    }
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory="[]",
        llama_stats=json.dumps(llama_stats),
        ollama_stats=None,
    )
    
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    
    # Verify the Last rate elements exist in the template
    assert 'id="tok-last-prompt-llama"' in html
    assert 'id="tok-last-gen-llama"' in html
    
    # Verify the JavaScript structure for tracking
    assert 'lastNonZeroRates' in html
    assert 'lastNonZeroTimes' in html
    assert 'updateLastRateDisplay' in html
    
    # Verify the interval exists and calls updateLastRateDisplay
    assert 'setInterval' in html
    assert 'updateLastRateDisplay' in html
    
    # Verify no cross-contamination: separate tracking for prompt and generate
    assert 'llama: { prompt: null, generate: null }' in html


@pytest.mark.asyncio
async def test_dashboard_no_cross_contamination_prompt_generate(test_client, test_db_path):
    """Test that prompt and generate rates are tracked separately without cross-contamination."""
    from monitor.store import insert_telemetry_sample
    import json
    
    # Insert telemetry with both prompt and generate rates
    llama_stats = {
        "prompt_tokens": 1000,
        "generated_tokens": 500,
        "prompt_tokens_rate": 12.3,
        "generated_tokens_rate": 8.7,
    }
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory="[]",
        llama_stats=json.dumps(llama_stats),
        ollama_stats=None,
    )
    
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    
    # Verify separate tracking for prompt and generate per server
    assert 'llama: { prompt: null, generate: null }' in html
    assert 'ollama: { prompt: null, generate: null }' in html
    
    # Verify that updateLastRateDisplay uses the metric type parameter to distinguish prompt vs generate
    assert "metric === 'prompt'" in html or "metric == 'prompt'" in html


def test_dashboard_rendered_javascript_syntax_valid():
    """Test that the dashboard template renders syntactically valid JavaScript."""
    import subprocess
    import re
    from fastapi.testclient import TestClient
    from monitor.web.routes import app
    import monitor.store
    import tempfile
    from pathlib import Path
    
    # Create a temp database
    temp_dir = Path(tempfile.mkdtemp())
    temp_db = temp_dir / "test_monitor.db"
    original_path = monitor.store.DB_PATH
    monitor.store.DB_PATH = temp_db
    
    try:
        # Initialize DB
        import asyncio
        asyncio.run(monitor.store.init_db())
        
        # Add some data to ensure template renders fully
        from monitor.store import insert_telemetry_sample
        import json
        
        async def setup():
            await insert_telemetry_sample(
                memory_total=32_000_000_000,
                memory_free=16_000_000_000,
                memory_used=16_000_000_000,
                gpu_util=45.0,
                gpu_temp=70.0,
                gpu_power=150.0,
                process_memory="[]",
                llama_stats=json.dumps({
                    "prompt_tokens": 1000,
                    "generated_tokens": 500,
                    "prompt_tokens_rate": 10.5,
                    "generated_tokens_rate": 5.2,
                }),
                ollama_stats=None,
            )
        
        asyncio.run(setup())
        
        # Get the rendered HTML
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        html = response.text
        
        # Extract the JavaScript content
        match = re.search(r'<script>([\s\S]*?)</script>', html)
        assert match is not None, "Expected <script> tag in dashboard HTML"
        js_content = match.group(1)
        
        # Replace Jinja2 template syntax with valid JavaScript
        def replace_jinja(match):
            content = match.group(1).strip()
            # Handle "{{ var or default }}" pattern
            if " or " in content:
                parts = content.split(" or ")
                var_name = parts[0].strip()
                default = parts[1].strip()
                return default
            return "null"
        
        js_content = re.sub(r'\{\{([^}]+)\}\}', replace_jinja, js_content)
        
        # Write the extracted JavaScript to a temp file
        import tempfile as tmp
        with tmp.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_content)
            temp_js_path = f.name
        
        try:
            # Run node --check to verify syntax
            result = subprocess.run(
                ['node', '--check', temp_js_path],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"JavaScript syntax check failed: {result.stderr}"
        finally:
            import os
            os.unlink(temp_js_path)
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_api_plots_history_returns_only_required_columns(test_client):
    """Test that /api/plots/history returns only columns needed for plots."""
    import json
    
    # Insert a sample with all columns
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory=json.dumps([{"pid": 1234, "name": "test", "rss": 1_000_000_000}]),
        gpu_process_memory=json.dumps([{"pid": 1234, "name": "test", "gpu_memory": 8_000_000_000}]),
        llama_stats=json.dumps({"generated_tokens_rate": 10.0, "prompt_tokens_rate": 5.0}),
        ollama_stats=json.dumps({"ollama_generated_tokens_rate": 15.0, "ollama_prompt_tokens_rate": 7.0})
    )
    
    response = test_client.get("/api/plots/history?limit=1")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    
    row = data[0]
    
    # Should have columns needed for plots
    assert "timestamp" in row
    assert "gpu_util" in row
    assert "gpu_temp" in row
    assert "gpu_power" in row
    assert "memory_used" in row
    assert "memory_total" in row
    assert "llama_stats" in row
    assert "ollama_stats" in row
    
    # Should NOT have large columns
    assert "process_memory" not in row
    assert "gpu_process_memory" not in row


@pytest.mark.asyncio
async def test_api_plots_history_payload_size_reduction(test_client):
    """Test that payload size is significantly reduced with column pruning."""
    import json
    
    # Insert sample with large columns
    large_process_memory = json.dumps([
        {"pid": i, "name": f"process_{i}", "rss": 1_000_000_000 + i * 100_000_000}
        for i in range(100)
    ])
    large_gpu_process_memory = json.dumps([
        {"pid": i, "name": f"gpu_process_{i}", "gpu_memory": 500_000_000 + i * 50_000_000}
        for i in range(100)
    ])
    
    await insert_telemetry_sample(
        memory_total=32_000_000_000,
        memory_free=16_000_000_000,
        memory_used=16_000_000_000,
        gpu_util=45.0,
        gpu_temp=70.0,
        gpu_power=150.0,
        process_memory=large_process_memory,
        gpu_process_memory=large_gpu_process_memory,
        llama_stats=json.dumps({"generated_tokens_rate": 10.0}),
        ollama_stats=json.dumps({"ollama_generated_tokens_rate": 15.0})
    )
    
    # Get via old endpoint (returns all columns)
    response_old = test_client.get("/api/telemetry/history?limit=1")
    assert response_old.status_code == 200
    old_data = response_old.json()
    old_size = len(json.dumps(old_data))
    
    # Get via new endpoint (returns only plot columns)
    response_new = test_client.get("/api/plots/history?limit=1")
    assert response_new.status_code == 200
    new_data = response_new.json()
    new_size = len(json.dumps(new_data))
    
    # New payload should be significantly smaller (no process_memory, gpu_process_memory)
    assert new_size < old_size * 0.5, f"Payload not reduced enough: {new_size} vs {old_size}"


@pytest.mark.asyncio
async def test_api_plots_history_empty_database(test_client):
    """Test that /api/plots/history returns empty list for empty database."""
    response = test_client.get("/api/plots/history?limit=100")
    assert response.status_code == 200
    data = response.json()
    assert data == []


@pytest.mark.asyncio
async def test_catalog_timeline_html_returns_200(test_client, test_db_path):
    """Test that /catalog/timeline returns 200 and renders timeline events."""
    from monitor.store import insert_benchmark_run
    
    # Insert a benchmark run
    await insert_benchmark_run(
        model_id=1,
        workload_type="standard",
        standard_run=True,
        model_name="test-model",
        server_type="llama-server",
        tags="standard",
        gen_tok_s=100.0,
        ttft=0.5,
    )
    
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    html = response.text
    
    # Check that the timeline page renders
    assert "Catalog: Timeline" in html
    assert "benchmark run" in html


@pytest.mark.asyncio
async def test_catalog_timeline_html_links_to_run_plot(test_client, test_db_path):
    """Test that /catalog/timeline includes links to individual run plots."""
    from monitor.store import insert_benchmark_run
    
    # Insert a benchmark run
    run_id = await insert_benchmark_run(
        model_id=1,
        workload_type="standard",
        standard_run=True,
        model_name="test-model",
        server_type="llama-server",
        tags="standard",
        gen_tok_s=100.0,
        ttft=0.5,
    )
    
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    html = response.text
    
    # Check that the timeline includes a link to the run plot
    assert f"/runs/{run_id}" in html


@pytest.mark.asyncio
async def test_run_plot_html_returns_200_with_charts(test_client, test_db_path):
    """Test that /runs/{run_id} returns 200 and contains expected chart containers."""
    from monitor.store import insert_benchmark_run
    import json
    
    # Insert a benchmark run with samples_during
    samples_during = json.dumps([
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "memory_used": {"memory_used": 16_000_000_000},
            "gpu": {"gpu_temp": 70.0, "gpu_util": 45.0, "gpu_power": 150.0},
        },
        {
            "timestamp": "2024-01-01T00:00:01+00:00",
            "memory_used": {"memory_used": 16_500_000_000},
            "gpu": {"gpu_temp": 72.0, "gpu_util": 50.0, "gpu_power": 155.0},
        },
    ])
    
    run_id = await insert_benchmark_run(
        model_id=1,
        workload_type="standard",
        standard_run=True,
        model_name="test-model",
        server_type="llama-server",
        tags="standard",
        total_time=10.0,
        gen_tok_s=100.0,
        ttft=0.5,
        samples_during=samples_during,
    )
    
    response = test_client.get(f"/runs/{run_id}")
    
    assert response.status_code == 200
    html = response.text
    
    # Check for expected chart containers
    assert 'id="memory-chart"' in html
    assert 'id="gpu-temp-chart"' in html
    assert 'id="gpu-util-chart"' in html
    assert 'id="gpu-power-chart"' in html
    assert 'id="tok-s-chart"' in html
    
    # Check that Back to Live View button exists
    assert "Back to Live View" in html
    assert "href=\"/plots\"" in html


@pytest.mark.asyncio
async def test_api_benchmarks_run_samples_returns_samples_during(test_client, test_db_path):
    """Test that /api/benchmarks/run/{run_id}/samples returns the samples_during."""
    from monitor.store import insert_benchmark_run
    import json
    
    # Insert a benchmark run with samples_during
    samples_during = json.dumps([
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "memory_used": {"memory_used": 16_000_000_000},
            "gpu": {"gpu_temp": 70.0, "gpu_util": 45.0, "gpu_power": 150.0},
            "llama_gen_rate": 100.0,
            "llama_prompt_rate": 10.0,
            "ollama_gen_rate": 50.0,
        },
        {
            "timestamp": "2024-01-01T00:00:01+00:00",
            "memory_used": {"memory_used": 16_500_000_000},
            "gpu": {"gpu_temp": 72.0, "gpu_util": 50.0, "gpu_power": 155.0},
            "llama_gen_rate": 105.0,
            "ollama_gen_rate": 55.0,
        },
    ])
    
    run_id = await insert_benchmark_run(
        model_id=1,
        workload_type="standard",
        standard_run=True,
        model_name="test-model",
        server_type="llama-server",
        tags="standard",
        total_time=10.0,
        gen_tok_s=100.0,
        ttft=0.5,
        samples_during=samples_during,
    )
    
    response = test_client.get(f"/api/benchmarks/run/{run_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    
    samples = data["samples"]
    assert len(samples) == 2
    
    # Verify tok/s data is present
    assert "llama_gen_rate" in samples[0]
    assert "llama_prompt_rate" in samples[0]
    assert "ollama_gen_rate" in samples[0]


@pytest.mark.asyncio
async def test_sample_during_run_captures_telemetry_and_tok_s(test_db_path):
    """Test that _sample_during_run samples telemetry at 1 Hz and captures tok/s rates.
    
    This exercises the sampling cadence (1 Hz), cancellation, and error tolerance.
    The test verifies:
    - samples are captured at ~1 second intervals
    - memory and GPU data are captured
    - tok/s rates (llama_gen_rate, llama_prompt_rate, ollama_gen_rate) are captured when available
    - cancellation stops sampling
    - telemetry errors don't crash the sampling task
    """
    import asyncio
    import monitor.store
    from monitor.benchmarks import BenchmarkRunner
    from monitor.telemetry.fixtures import (
        FixtureMemorySource, FixtureGPUSource, FixtureTelemetrySource,
        FixtureLLaMAStatsSource, FixtureOllamaStatsSource
    )
    
    monitor.store.DB_PATH = test_db_path
    
    # Create a runner with fixture telemetry sources that provide tok/s data
    runner = BenchmarkRunner()
    
    # Override telemetry sources with fixture sources that have tok/s rates
    runner._memory_source = FixtureMemorySource(
        total=32_000_000_000,
        free=16_000_000_000,
        used=16_000_000_000
    )
    runner._gpu_source = FixtureGPUSource(
        util=45.0,
        temp=70.0,
        power=150.0
    )
    runner._llama_stats_source = FixtureLLaMAStatsSource(
        prompt_tokens=10000,
        generated_tokens=5000,
        speculative_accepts=500,
        prompt_tokens_rate=10.0,
        generated_tokens_rate=100.0,
        speculative_accepts_rate=0.5
    )
    runner._ollama_stats_source = FixtureOllamaStatsSource(
        prompt_tokens=8000,
        generated_tokens=4000,
        prompt_tokens_rate=8.0,
        generated_tokens_rate=50.0
    )
    
    # Collect samples for 3 seconds
    samples_during = []
    sample_task = asyncio.create_task(runner._sample_during_run(samples_during))
    
    # Wait for ~3 samples
    await asyncio.sleep(3.5)
    
    # Cancel the sampling task
    sample_task.cancel()
    try:
        await sample_task
    except asyncio.CancelledError:
        pass
    
    # Verify we got samples
    assert len(samples_during) >= 2, f"Expected at least 2 samples, got {len(samples_during)}"
    
    # Verify first sample structure
    first_sample = samples_during[0]
    assert "timestamp" in first_sample
    assert "memory_used" in first_sample
    assert "gpu" in first_sample
    
    # Verify memory data
    assert "memory_used" in first_sample["memory_used"]
    
    # Verify GPU data
    gpu = first_sample["gpu"]
    assert "gpu_util" in gpu
    assert "gpu_temp" in gpu
    assert "gpu_power" in gpu
    
    # Verify llama tok/s data (should be captured)
    assert "llama_gen_rate" in first_sample
    assert "llama_prompt_rate" in first_sample
    
    # Verify ollama tok/s data (should be captured)
    assert "ollama_gen_rate" in first_sample
    assert "ollama_prompt_rate" in first_sample
    
    # Verify values match fixture sources
    assert first_sample["llama_gen_rate"] == 100.0
    assert first_sample["llama_prompt_rate"] == 10.0
    assert first_sample["ollama_gen_rate"] == 50.0
    assert first_sample["ollama_prompt_rate"] == 8.0
    
    # Verify samples are taken at approximately 1 second intervals
    # (We expect ~3 samples in 3.5 seconds)
    for i in range(1, len(samples_during)):
        prev_ts = first_sample["timestamp"] if i == 1 else samples_during[i-1]["timestamp"]
        # Parse ISO format timestamps
        from datetime import datetime
        prev_dt = datetime.fromisoformat(prev_ts.replace('Z', '+00:00'))
        curr_dt = datetime.fromisoformat(samples_during[i]["timestamp"].replace('Z', '+00:00'))
        delta = (curr_dt - prev_dt).total_seconds()
        assert 0.8 <= delta <= 1.5, f"Sample interval should be ~1s, got {delta}s"


@pytest.mark.asyncio
async def test_sample_during_run_error_tolerance(test_db_path):
    """Test that _sample_during_run continues sampling despite telemetry errors."""
    import asyncio
    import monitor.store
    from monitor.benchmarks import BenchmarkRunner
    from monitor.telemetry.fixtures import FixtureMemorySource, FixtureGPUSource, FixtureOllamaStatsSource
    
    monitor.store.DB_PATH = test_db_path
    
    # Create a runner with fixture sources
    runner = BenchmarkRunner()
    
    # Use fixture sources that work
    runner._memory_source = FixtureMemorySource(
        total=32_000_000_000,
        free=16_000_000_000,
        used=16_000_000_000
    )
    runner._gpu_source = FixtureGPUSource(
        util=45.0,
        temp=70.0,
        power=150.0
    )
    
    # Mock tok/s sources to raise an error on first call, then succeed
    call_count = 0
    
    class ErrorThenSuccessLLaMAStats:
        async def collect(self):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Simulated telemetry error")
            return {
                "prompt_tokens": 10000,
                "generated_tokens": 5000,
                "prompt_tokens_rate": 10.0,
                "generated_tokens_rate": 100.0,
            }
    
    runner._llama_stats_source = ErrorThenSuccessLLaMAStats()
    runner._ollama_stats_source = FixtureOllamaStatsSource(
        prompt_tokens=8000,
        generated_tokens=4000,
        prompt_tokens_rate=8.0,
        generated_tokens_rate=50.0
    )
    
    # Collect samples for 2 seconds
    samples_during = []
    sample_task = asyncio.create_task(runner._sample_during_run(samples_during))
    
    await asyncio.sleep(2.5)
    
    sample_task.cancel()
    try:
        await sample_task
    except asyncio.CancelledError:
        pass
    
    # Verify we still got samples despite the error
    assert len(samples_during) >= 1, f"Expected at least 1 sample, got {len(samples_during)}"
    
    # Verify first sample has the tok/s rate from the working source
    # (ollama is still working, llama errors are handled gracefully)
    assert "ollama_gen_rate" in samples_during[0]
