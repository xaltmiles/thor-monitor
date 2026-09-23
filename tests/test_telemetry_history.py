"""Tests for telemetry history optimization (issue #28)."""

import pytest
import asyncio
import json
from pathlib import Path
from monitor.store import (
    init_db, insert_telemetry_sample, get_telemetry_history, DB_PATH
)


@pytest.mark.asyncio
async def test_index_on_timestamp_exists(temp_db_path):
    """Test that an index exists on telemetry_samples.timestamp column."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Check that index exists on timestamp column
        async with monitor.store.aiosqlite.connect(temp_db_path) as db:
            # Get all indexes for telemetry_samples table
            cursor = await db.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='telemetry_samples'
            """)
            indexes = await cursor.fetchall()
            index_names = [idx[0] for idx in indexes]
            
            # There should be an index on timestamp
            timestamp_idx = [idx for idx in index_names if 'timestamp' in idx.lower()]
            assert len(timestamp_idx) > 0, f"Expected index on timestamp, found: {index_names}"
            
            # Verify the index includes timestamp column
            for idx_name in timestamp_idx:
                cursor = await db.execute(f"PRAGMA index_info({idx_name})")
                info = await cursor.fetchall()
                # First column should be timestamp
                if info:
                    assert info[0][2] == 'timestamp', f"Index {idx_name} doesn't start with timestamp"
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_get_telemetry_history_for_plots_returns_only_required_columns(temp_db_path):
    """Test that get_telemetry_history_for_plots returns only columns needed for plots."""
    from monitor.store import DB_PATH as original_path, get_telemetry_history_for_plots
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
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
        
        # Retrieve history
        history = await get_telemetry_history_for_plots(limit=1)
        
        assert len(history) == 1
        row = history[0]
        
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
        
        # Values should be correct
        assert row["gpu_util"] == 45.0
        assert row["gpu_temp"] == 70.0
        assert row["gpu_power"] == 150.0
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_get_telemetry_history_performance_with_index(temp_db_path):
    """Test that history queries use the index and are fast."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    import time
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Insert 1000 samples (simulating 1 hour at 6s intervals)
        for i in range(1000):
            await insert_telemetry_sample(
                memory_total=32_000_000_000,
                memory_free=16_000_000_000,
                memory_used=16_000_000_000,
                gpu_util=45.0 + (i % 50),
                gpu_temp=70.0 + (i % 15),
                gpu_power=150.0 + (i % 30),
                process_memory="[]",
                gpu_process_memory="[]",
                llama_stats=json.dumps({"generated_tokens_rate": 10.0 + i * 0.01}),
                ollama_stats=json.dumps({"ollama_generated_tokens_rate": 15.0 + i * 0.01})
            )
        
        # Query last 3600 samples (1 hour)
        start = time.time()
        history = await get_telemetry_history(limit=3600)
        elapsed = time.time() - start
        
        assert len(history) <= 3600
        # Should complete in under 1 second (actual should be much faster)
        assert elapsed < 1.0, f"Query took {elapsed:.2f}s, should be < 1s"
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_telemetry_history_empty_database(temp_db_path):
    """Test that get_telemetry_history returns empty list for empty database."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        history = await get_telemetry_history(limit=100)
        assert history == []
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_telemetry_history_ordering(temp_db_path):
    """Test that history is returned in correct order (newest first)."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    import time
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Insert samples with known timestamps
        base_time = "2024-01-01T00:00:00+00:00"
        
        for i in range(5):
            # Convert ISO format to SQLite-compatible format
            ts = f"2024-01-01T00:{i:02d}:00+00:00"
            await insert_telemetry_sample(
                memory_total=32_000_000_000,
                memory_free=16_000_000_000,
                memory_used=16_000_000_000 + i * 1_000_000,
                gpu_util=45.0 + i,
                gpu_temp=70.0 + i,
                gpu_power=150.0 + i,
                process_memory="[]",
                gpu_process_memory="[]",
                llama_stats="[]",
                ollama_stats="[]"
            )
            # Wait a bit to ensure different timestamps
            await asyncio.sleep(0.01)
        
        history = await get_telemetry_history(limit=5)
        
        # Should be ordered newest first
        for i in range(len(history) - 1):
            # Compare timestamps (newest first)
            assert history[i]["timestamp"] >= history[i + 1]["timestamp"]
    finally:
        monitor.store.DB_PATH = original_path
