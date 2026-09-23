"""Tests for server-type-specific catalog views (issue #25)."""

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
    asyncio.run(init_db())
    return TestClient(app)


@pytest.fixture
def seeded_db_same_model_different_servers(test_db_path):
    """Create a database with the same model benchmarked via different servers.
    
    This is the key scenario for issue #25: same model name, but different
    inference servers (Ollama vs llama.cpp/unsloth).
    """
    import asyncio
    from datetime import datetime, timezone
    
    asyncio.run(init_db())
    
    # Insert model metadata for both server types
    # In reality these would be separate entries in models table
    # but for testing purposes we'll use the same model_id with different server_types
    # Note: the actual schema stores server_type in benchmark_runs, not models
    # So we need separate model entries if server_type is in models
    
    # Model benchmarked via Ollama
    model_ollama_id = asyncio.run(insert_model(
        name="Llama-3.1-8B",
        quant="Instruct",
        context_length=128_000,
        server_type="ollama",
        file_size=4_800_000_000
    ))
    
    # Same model benchmarked via llama.cpp (unsloth)
    model_llama_id = asyncio.run(insert_model(
        name="Llama-3.1-8B",
        quant="Instruct",
        context_length=128_000,
        server_type="llama-server",
        file_size=4_800_000_000
    ))
    
    # Benchmark run via Ollama
    asyncio.run(insert_benchmark_run(
        model_id=model_ollama_id,
        model_name="Llama-3.1-8B",
        model_quant="Instruct",
        context_length=128_000,
        server_type="ollama",
        server_port=11434,
        workload_type="short",
        standard_run=True,
        total_time=15.0,
        ttft=0.35,
        prompt_tok_s=42.0,
        gen_tok_s=45.0,
        peak_gen_tok_s=55.0,
        concurrent_throughput=None,
        tags="standard,short-workload"
    ))
    
    # Same model benchmarked via llama.cpp (unsloth) - different server, different speed
    asyncio.run(insert_benchmark_run(
        model_id=model_llama_id,
        model_name="Llama-3.1-8B",
        model_quant="Instruct",
        context_length=128_000,
        server_type="llama-server",
        server_port=34329,
        workload_type="short",
        standard_run=True,
        total_time=10.0,
        ttft=0.20,
        prompt_tok_s=58.0,
        gen_tok_s=62.0,
        peak_gen_tok_s=75.0,
        concurrent_throughput=None,
        tags="standard,short-workload"
    ))
    
    # Another workload for Ollama
    asyncio.run(insert_benchmark_run(
        model_id=model_ollama_id,
        model_name="Llama-3.1-8B",
        model_quant="Instruct",
        context_length=128_000,
        server_type="ollama",
        server_port=11434,
        workload_type="long-context",
        standard_run=True,
        total_time=60.0,
        ttft=None,
        prompt_tok_s=35.0,
        gen_tok_s=None,
        peak_gen_tok_s=None,
        concurrent_throughput=None,
        tags="standard,long-context"
    ))
    
    # Session for Ollama run
    asyncio.run(insert_session(
        model_id=model_ollama_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        avg_tok_s=43.0,
        total_tokens=8000
    ))
    
    # Session for llama.cpp run
    asyncio.run(insert_session(
        model_id=model_llama_id,
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        avg_tok_s=60.0,
        total_tokens=12000
    ))
    
    return test_db_path


class TestCatalogComparisonServerType:
    """Test that catalog comparison groups by server type (issue #25)."""
    
    @pytest.mark.asyncio
    async def test_comparison_shows_separate_rows_for_different_servers(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """Same model benchmarked on different servers should appear as separate rows."""
        response = test_client.get("/catalog/comparison")
        assert response.status_code == 200
        
        response_text = response.text
        
        # Should show "Llama-3.1-8B" appearing multiple times (once per server)
        # Count occurrences - should be at least 2 (one per server type)
        import re
        model_matches = re.findall(r'>Llama-3\.1-8B</strong>', response_text)
        assert len(model_matches) >= 2, \
            f"Expected at least 2 rows for Llama-3.1-8B (one per server), found {len(model_matches)}"
        
        # Should have separate Server column values
        # Count how many times each server type appears
        ollama_matches = re.findall(r'>ollama</td>', response_text)
        llama_matches = re.findall(r'>llama-server</td>', response_text)
        
        assert len(ollama_matches) >= 1, "Should have at least one row with Ollama server"
        assert len(llama_matches) >= 1, "Should have at least one row with llama-server"
    
    @pytest.mark.asyncio
    async def test_comparison_shows_different_tok_s_per_server(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """Each server should show its own tok/s metrics."""
        response = test_client.get("/catalog/comparison")
        assert response.status_code == 200
        
        response_text = response.text
        
        # Ollama run has gen_tok_s=45.0
        # llama.cpp run has gen_tok_s=62.0
        # Both should appear in the comparison
        
        # Check for the llama.cpp speed (62.0) - should be present
        assert "62.0" in response_text, \
            "llama.cpp server's 62.0 tok/s should be present in comparison"
        
        # Check that Ollama speed (45.0) is also present
        assert "45.0" in response_text, \
            "Ollama server's 45.0 tok/s should be present in comparison"
        
        # The values should be different (proving they're not averaged)
        # Count how many times each value appears
        import re
        # Find all tok/s values in the table body
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', response_text, re.DOTALL)
        if tbody_match:
            tbody = tbody_match.group(1)
            # Both 45.0 and 62.0 should appear in the tbody
            assert "45.0" in tbody, "45.0 tok/s (Ollama) should be in table body"
            assert "62.0" in tbody, "62.0 tok/s (llama.cpp) should be in table body"
    
    @pytest.mark.asyncio
    async def test_comparison_shows_server_type_column(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """Server column should show the correct server type for each row."""
        response = test_client.get("/catalog/comparison")
        assert response.status_code == 200
        
        response_text = response.text
        
        # The Server column should contain the server types
        # Check for the server column content
        import re
        # Look for the Server column values in table cells
        server_matches = re.findall(r'<td>(ollama|llama-server)</td>', response_text)
        
        assert len(server_matches) >= 2, \
            f"Should have at least 2 server type entries, found {len(server_matches)}"
        assert "ollama" in server_matches, "Ollama server type should be present"
        assert "llama-server" in server_matches, "llama-server type should be present"


class TestCatalogTimelineServerType:
    """Test that timeline includes server type in event details (issue #25)."""
    
    @pytest.mark.asyncio
    async def test_timeline_shows_server_type_in_benchmark_details(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """Benchmark event details should include which server produced the speed."""
        response = test_client.get("/catalog/timeline")
        assert response.status_code == 200
        
        response_text = response.text
        
        # Timeline should show server type info
        # The details string should include "llama-server" or "ollama" context
        
        # Check for server type indicators in the timeline
        # For llama.cpp run, details should mention "llama" or the server type
        assert "llama" in response_text.lower() or "llama-server" in response_text.lower(), \
            "Timeline should show llama server type info"
        
        # Check for the specific tok/s values from different servers
        assert "62.0" in response_text, \
            "Timeline should show llama.cpp server's 62.0 tok/s"
    
    @pytest.mark.asyncio
    async def test_timeline_shows_server_type_for_sessions(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """Session events should also show server type."""
        response = test_client.get("/catalog/timeline")
        assert response.status_code == 200
        
        response_text = response.text
        
        # Sessions should appear in the timeline
        assert "session" in response_text.lower(), \
            "Timeline should contain session entries"
        
        # Should show different speeds for different servers
        # Ollama session: avg_tok_s=43.0
        # llama.cpp session: avg_tok_s=60.0
        assert "43.0" in response_text or "42.0" in response_text or "45.0" in response_text, \
            "Timeline should show Ollama session tok/s"
        assert "60.0" in response_text or "62.0" in response_text, \
            "Timeline should show llama.cpp session tok/s"


class TestPlotsHtmlLabel:
    """Test that plots.html uses correct labels for tok/s lines (issue #25)."""
    
    @pytest.mark.asyncio
    async def test_plots_uses_llama_cpp_label_not_llama(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """The tok/s chart legend should say 'llama.cpp tok/s', not 'Llama tok/s'."""
        response = test_client.get("/plots")
        assert response.status_code == 200
        
        response_text = response.text
        
        # Should NOT contain the old misleading label
        assert "Llama tok/s" not in response_text, \
            "Plots page should not use 'Llama tok/s' label (ambiguous)"
        
        # Should contain the clearer "llama.cpp" label
        assert "llama.cpp tok/s" in response_text, \
            "Plots page should use 'llama.cpp tok/s' label"


class TestServerTypeDataStructure:
    """Test that server_type is properly stored and retrieved."""
    
    @pytest.mark.asyncio
    async def test_server_type_stored_in_benchmark_runs(
        self, test_client, seeded_db_same_model_different_servers
    ):
        """server_type should be stored in benchmark runs table."""
        response = test_client.get("/catalog/comparison")
        assert response.status_code == 200
        
        # The response should show different server types for same model
        response_text = response.text
        
        # Should see the Server column with different values
        import re
        server_cells = re.findall(r'<td>\s*(ollama|llama-server)\s*</td>', response_text)
        
        # Both server types should be present
        assert len(server_cells) >= 2, \
            f"Expected at least 2 server type cells, found {len(server_cells)}"
        assert set(server_cells) == {"ollama", "llama-server"}, \
            f"Expected both server types, found {set(server_cells)}"
