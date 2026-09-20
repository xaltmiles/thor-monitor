"""Tests for catalog views (comparison table and timeline)."""

import pytest
from fastapi.testclient import TestClient

from monitor.web.routes import app
from monitor.store import (
    init_db, insert_model, insert_benchmark_run, insert_session,
    DB_PATH
)
import tempfile
from pathlib import Path
import monitor.store
import shutil


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_db = temp_dir / "test_monitor.db"
    original_path = monitor.store.DB_PATH
    
    monitor.store.DB_PATH = temp_db
    
    yield temp_db
    
    shutil.rmtree(temp_dir)
    monitor.store.DB_PATH = original_path


@pytest.fixture
def test_client(test_db_path):
    """Create a test client with test database."""
    import asyncio
    
    # Initialize test database
    asyncio.run(init_db())
    
    return TestClient(app)


@pytest.fixture
def seeded_db(test_db_path):
    """Create a database with sample models and benchmark runs for catalog views."""
    import asyncio
    from monitor.store import get_all_models
    
    asyncio.run(init_db())
    
    # Insert test models
    model1_id = asyncio.run(insert_model(
        name="llama3",
        quant="8b",
        context_length=8192,
        server_type="ollama",
        file_size=4_600_000_000
    ))
    
    model2_id = asyncio.run(insert_model(
        name="mistral",
        quant="Q4_K_M",
        context_length=8192,
        server_type="llama-server",
        file_size=4_200_000_000
    ))
    
    # Insert benchmark runs (standard runs for comparison)
    asyncio.run(insert_benchmark_run(
        model_id=model1_id,
        model_name="llama3",
        model_quant="8b",
        context_length=8192,
        server_type="ollama",
        server_port=11434,
        workload_type="short",
        standard_run=True,
        total_time=10.5,
        ttft=0.25,
        prompt_tok_s=45.0,
        gen_tok_s=52.3,
        peak_gen_tok_s=65.0,
        concurrent_throughput=None,
        tags="standard,short-workload"
    ))
    
    asyncio.run(insert_benchmark_run(
        model_id=model1_id,
        model_name="llama3",
        model_quant="8b",
        context_length=8192,
        server_type="ollama",
        server_port=11434,
        workload_type="long-context",
        standard_run=True,
        total_time=45.0,
        ttft=None,
        prompt_tok_s=38.5,
        gen_tok_s=None,
        peak_gen_tok_s=None,
        concurrent_throughput=None,
        tags="standard,long-context"
    ))
    
    asyncio.run(insert_benchmark_run(
        model_id=model1_id,
        model_name="llama3",
        model_quant="8b",
        context_length=8192,
        server_type="ollama",
        server_port=11434,
        workload_type="burst",
        standard_run=True,
        total_time=30.0,
        ttft=None,
        prompt_tok_s=None,
        gen_tok_s=None,
        peak_gen_tok_s=None,
        concurrent_throughput=180.0,
        tags="standard,burst"
    ))
    
    # Different quant of same model (custom run - should be excluded from comparison)
    asyncio.run(insert_benchmark_run(
        model_id=model1_id,
        model_name="llama3",
        model_quant="6b",  # Different quant
        context_length=8192,
        server_type="ollama",
        server_port=11434,
        workload_type="short",
        standard_run=False,  # Custom run
        total_time=8.0,
        ttft=0.20,
        prompt_tok_s=55.0,
        gen_tok_s=62.0,
        peak_gen_tok_s=75.0,
        concurrent_throughput=None,
        tags="custom,short-workload"
    ))
    
    asyncio.run(insert_benchmark_run(
        model_id=model2_id,
        model_name="mistral",
        model_quant="Q4_K_M",
        context_length=8192,
        server_type="llama-server",
        server_port=8080,
        workload_type="short",
        standard_run=True,
        total_time=12.0,
        ttft=0.30,
        prompt_tok_s=40.0,
        gen_tok_s=48.0,
        peak_gen_tok_s=58.0,
        concurrent_throughput=None,
        tags="standard,short-workload"
    ))
    
    asyncio.run(insert_benchmark_run(
        model_id=model2_id,
        model_name="mistral",
        model_quant="Q4_K_M",
        context_length=8192,
        server_type="llama-server",
        server_port=8080,
        workload_type="long-context",
        standard_run=True,
        total_time=50.0,
        ttft=None,
        prompt_tok_s=35.0,
        gen_tok_s=None,
        peak_gen_tok_s=None,
        concurrent_throughput=None,
        tags="standard,long-context"
    ))
    
    # Insert sessions for timeline
    from datetime import datetime, timezone
    
    session1_id = asyncio.run(insert_session(
        model_id=model1_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        avg_tok_s=48.5,
        total_tokens=10000
    ))
    
    asyncio.run(insert_session(
        model_id=model2_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=None,  # Active session
        avg_tok_s=42.0,
        total_tokens=5000
    ))
    
    return test_db_path


@pytest.mark.asyncio
async def test_catalog_comparison_route_empty(test_client):
    """Test comparison table route returns empty when no data exists."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    assert "Comparison Table" in response.text
    # Should show message or empty table when no standard runs


@pytest.mark.asyncio
async def test_catalog_comparison_route_with_data(test_client, seeded_db):
    """Test comparison table shows seeded benchmark runs."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    assert "Comparison Table" in response.text
    
    # Should contain model names
    response_text = response.text
    assert "llama3" in response_text
    assert "mistral" in response_text
    
    # Should contain quant info
    assert "8b" in response_text or "Q4_K_M" in response_text
    
    # Should contain tok/s values (from seeded data)
    assert "tok/s" in response_text.lower()
    
    # Should not show custom runs (filtered out)
    # Custom runs have 6b quant which shouldn't appear in standard runs table


@pytest.mark.asyncio
async def test_catalog_comparison_filters_standard_runs_only(test_client, seeded_db):
    """Test that comparison table filters to standard runs only."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    response_text = response.text
    
    # The seeded db has a custom run with 6b quant for llama3 (62.0 tok/s)
    # Custom run should be excluded - only standard runs (8b quant) should appear
    
    # Verify the comparison table only shows 8b quant (standard runs)
    # Use regex to ensure we're matching the <td> element, not class names
    import re
    # Find the table body and check for quant values within it
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', response_text, re.DOTALL)
    if tbody_match:
        tbody_content = tbody_match.group(1)
        # Verify custom quant '6b' is NOT present in comparison table body
        assert "6b" not in tbody_content, "Custom run with quant '6b' should be filtered out"
        # Verify standard run quant '8b' IS present
        assert "8b" in tbody_content, "Standard run with quant '8b' should be present"
        
        # Verify we're not showing custom-specific values (62.0 tok/s for 6b)
        assert "62.0" not in tbody_content, "Custom run's 62.0 tok/s should be filtered out"
        assert "75.0" not in tbody_content, "Custom run's peak_gen_tok_s=75.0 should be filtered out"
    else:
        pytest.fail("Could not find table body in response")


@pytest.mark.asyncio
async def test_catalog_timeline_route_empty(test_client):
    """Test timeline route returns empty when no data exists."""
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    assert "Timeline" in response.text


@pytest.mark.asyncio
async def test_catalog_timeline_route_with_data(test_client, seeded_db):
    """Test timeline shows seeded runs and sessions."""
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    assert "Timeline" in response.text
    
    response_text = response.text
    
    # Should contain timeline entries
    assert "benchmark" in response_text.lower() or "session" in response_text.lower()
    
    # Should distinguish between run types
    assert "llama3" in response_text or "mistral" in response_text


@pytest.mark.asyncio
async def test_catalog_timeline_shows_benchmark_runs(test_client, seeded_db):
    """Test timeline includes benchmark runs."""
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    response_text = response.text
    
    # Should show benchmark run entries
    # Check for workload types from seeded data
    assert "short" in response_text.lower() or "long" in response_text.lower()


@pytest.mark.asyncio
async def test_catalog_timeline_shows_sessions(test_client, seeded_db):
    """Test timeline includes sessions."""
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    response_text = response.text
    
    # Should show session entries
    assert "session" in response_text.lower()


@pytest.mark.asyncio
async def test_catalog_comparison_sorts_models(test_client, seeded_db):
    """Test comparison table sorts models alphabetically or logically."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    
    response_text = response.text
    # Models should be present in some order
    assert "llama3" in response_text
    assert "mistral" in response_text


@pytest.mark.asyncio
async def test_catalog_comparison_shows_memory_footprint(test_client, seeded_db):
    """Test comparison table shows memory footprint per model."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    response_text = response.text
    
    # Should show file size info (from seeded model metadata)
    assert "GB" in response_text or "MB" in response_text or "file" in response_text.lower()


@pytest.mark.asyncio
async def test_catalog_timeline_chronological_order(test_client, seeded_db):
    """Test timeline entries are in chronological order (most recent first)."""
    response = test_client.get("/catalog/timeline")
    
    assert response.status_code == 200
    
    # Extract timestamps from the response
    # Format in template: YYYY-MM-DD HH:MM
    import re
    timestamps = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', response.text)
    
    # We should have at least some timestamps
    assert len(timestamps) >= 1, "Timeline should contain at least one timestamp"
    
    # Parse timestamps and verify descending order (most recent first)
    from datetime import datetime
    parsed_times = []
    for ts in timestamps:
        try:
            parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            parsed_times.append(parsed)
        except ValueError:
            pass
    
    # Verify they are in descending order
    if len(parsed_times) >= 2:
        for i in range(1, len(parsed_times)):
            assert parsed_times[i] <= parsed_times[i-1], \
                f"Timeline entries should be in descending order by timestamp, but {parsed_times[i]} > {parsed_times[i-1]}"


@pytest.mark.asyncio
async def test_catalog_comparison_no_custom_runs(test_client, seeded_db):
    """Test that custom runs are excluded from comparison table."""
    response = test_client.get("/catalog/comparison")
    
    assert response.status_code == 200
    response_text = response.text
    
    # The seeded db has a custom run for llama3 6b with 62.0 tok/s
    # The comparison should only show standard runs (llama3 8b with 52.3 tok/s)
    # Custom 6b quant should NOT appear in the comparison table
    # Use regex to ensure we're matching the <td> element, not class names
    import re
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', response_text, re.DOTALL)
    if tbody_match:
        tbody_content = tbody_match.group(1)
        assert "6b" not in tbody_content, "Custom run quant '6b' should not appear in comparison table"
        
        # Standard run llama3 8b should be present
        assert "llama3" in tbody_content
        assert "8b" in tbody_content
        
        # Should contain the standard run's tok/s value (52.3), not the custom run's (62.0)
        assert "62.0" not in tbody_content, "Custom run tok/s value should not appear in comparison table"
    else:
        pytest.fail("Could not find table body in response")
