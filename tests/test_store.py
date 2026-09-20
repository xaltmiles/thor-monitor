"""Tests for store module."""

import pytest
import asyncio
from pathlib import Path

from monitor.store import (
    init_db, insert_telemetry_sample, get_latest_telemetry, 
    get_telemetry_history, DB_PATH
)
from monitor.telemetry.fixtures import FixtureMemorySource, FixtureGPUSource, FixtureTelemetrySource


@pytest.mark.asyncio
async def test_init_db_creates_tables(temp_db_path):
    """Test that database initialization creates all required tables."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Verify tables exist
        async with monitor.store.aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]
            
            assert "telemetry_samples" in table_names
            assert "models" in table_names
            assert "benchmark_runs" in table_names
            assert "sessions" in table_names
            assert "settings" in table_names
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_insert_and_get_telemetry_sample(temp_db_path):
    """Test inserting and retrieving telemetry samples."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Insert a sample
        sample = await insert_telemetry_sample(
            memory_total=32_000_000_000,
            memory_free=16_000_000_000,
            memory_used=16_000_000_000,
            gpu_util=45.0,
            gpu_temp=70.0,
            gpu_power=150.0,
            process_memory="[]"
        )
        
        # Retrieve it
        latest = await get_latest_telemetry()
        
        assert latest is not None
        assert latest["memory_total"] == 32_000_000_000
        assert latest["memory_free"] == 16_000_000_000
        assert latest["gpu_util"] == 45.0
        assert latest["gpu_temp"] == 70.0
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_get_telemetry_history(temp_db_path):
    """Test retrieving telemetry history."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Insert multiple samples
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
        
        # Retrieve history
        history = await get_telemetry_history(limit=3)
        
        assert len(history) == 3
        assert history[0]["memory_used"] == 16_000_000_000 + 4 * 1_000_000_000  # Most recent
        assert history[1]["memory_used"] == 16_000_000_000 + 3 * 1_000_000_000
        assert history[2]["memory_used"] == 16_000_000_000 + 2 * 1_000_000_000
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_insert_telemetry_sample_with_gpu_process_memory(temp_db_path):
    """Test inserting telemetry sample with GPU process memory attribution."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    import json
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Insert a sample with GPU process memory
        gpu_procs = [
            {"pid": 1234, "name": "ollama", "gpu_memory": 8_000_000_000},
            {"pid": 5678, "name": "llama-server", "gpu_memory": 12_000_000_000},
        ]
        
        sample = await insert_telemetry_sample(
            memory_total=32_000_000_000,
            memory_free=16_000_000_000,
            memory_used=16_000_000_000,
            gpu_util=45.0,
            gpu_temp=70.0,
            gpu_power=150.0,
            process_memory="[]",
            gpu_process_memory=json.dumps(gpu_procs)
        )
        
        # Retrieve it
        latest = await get_latest_telemetry()
        
        assert latest is not None
        assert latest["gpu_process_memory"] is not None
        
        # Parse and verify GPU process data
        retrieved_procs = json.loads(latest["gpu_process_memory"])
        assert len(retrieved_procs) == 2
        assert retrieved_procs[0]["name"] == "ollama"
        assert retrieved_procs[0]["gpu_memory"] == 8_000_000_000
    finally:
        monitor.store.DB_PATH = original_path


@pytest.mark.asyncio
async def test_init_db_creates_tables_with_gpu_process_memory(temp_db_path):
    """Test that init_db creates table with gpu_process_memory column."""
    from monitor.store import DB_PATH as original_path
    import monitor.store
    monitor.store.DB_PATH = temp_db_path
    
    try:
        await init_db()
        
        # Verify gpu_process_memory column exists
        async with monitor.store.aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute("PRAGMA table_info(telemetry_samples)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            assert "gpu_process_memory" in column_names
    finally:
        monitor.store.DB_PATH = original_path
