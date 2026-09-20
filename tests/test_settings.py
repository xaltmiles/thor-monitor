"""Tests for settings functionality."""

import pytest
import asyncio
import json
from pathlib import Path

from monitor.store import DB_PATH, init_db, get_settings, update_settings


class TestSettingsStore:
    """Tests for settings store functions."""
    
    @pytest.mark.asyncio
    async def test_settings_table_exists(self, temp_db_path):
        """Test that settings table is created during init_db."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            # Verify settings table exists
            async with monitor.store.aiosqlite.connect(temp_db_path) as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
                )
                table = await cursor.fetchone()
                assert table is not None
                
                # Verify columns exist
                cursor = await db.execute("PRAGMA table_info(settings)")
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                assert "sample_rate" in column_names
                assert "standard_workload_duration" in column_names
                assert "max_queue_wait" in column_names
                assert "warning_gpu_temp" in column_names
                assert "warning_gpu_util" in column_names
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_get_default_settings(self, temp_db_path):
        """Test retrieving default settings."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            settings = await get_settings()
            
            assert settings is not None
            assert settings["sample_rate"] == 1
            assert settings["standard_workload_duration"] == 10
            assert settings["max_queue_wait"] == 120
            assert settings["warning_gpu_temp"] == 85.0
            assert settings["warning_gpu_util"] == 95.0
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_update_settings(self, temp_db_path):
        """Test updating settings."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            # Update settings
            updated = await update_settings(
                sample_rate=2,
                standard_workload_duration=20,
                max_queue_wait=60,
                warning_gpu_temp=80.0,
                warning_gpu_util=90.0
            )
            
            assert updated is not None
            assert updated["sample_rate"] == 2
            assert updated["standard_workload_duration"] == 20
            assert updated["max_queue_wait"] == 60
            assert updated["warning_gpu_temp"] == 80.0
            assert updated["warning_gpu_util"] == 90.0
            
            # Retrieve and verify persistence
            settings = await get_settings()
            
            assert settings["sample_rate"] == 2
            assert settings["standard_workload_duration"] == 20
            assert settings["max_queue_wait"] == 60
            assert settings["warning_gpu_temp"] == 80.0
            assert settings["warning_gpu_util"] == 90.0
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_update_settings_partial(self, temp_db_path):
        """Test updating only some settings."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            # Update only sample_rate
            updated = await update_settings(sample_rate=3)
            
            assert updated["sample_rate"] == 3
            
            # Other settings should remain default
            assert updated["standard_workload_duration"] == 10
            assert updated["max_queue_wait"] == 120
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_standard_custom_boundary_with_duration(self, temp_db_path):
        """Test that modifying standard_workload_duration changes the standard/custom boundary."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            # First, use default settings - max_tokens=512 for 10s duration
            settings = await get_settings()
            assert settings["standard_workload_duration"] == 10
            
            # Update to 20 seconds - max_tokens should scale to 1024
            updated = await update_settings(standard_workload_duration=20)
            assert updated["standard_workload_duration"] == 20
            
            # Now a run with max_tokens=1024 should be standard
            from monitor.benchmarks import BenchmarkRunner
            runner = BenchmarkRunner()
            
            # This should be standard now (20s duration -> 1024 tokens)
            is_standard = await runner._is_standard_run(max_tokens=1024, prompt_tokens=64, tag="standard")
            assert is_standard is True
            
            # And max_tokens=512 should be custom
            is_standard = await runner._is_standard_run(max_tokens=512, prompt_tokens=64, tag="standard")
            assert is_standard is False
        finally:
            monitor.store.DB_PATH = original_path


class TestSettingsAPI:
    """Tests for settings HTTP API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_settings_endpoint(self, temp_db_path):
        """Test GET /api/settings returns stored settings."""
        import monitor.store
        from monitor.web.routes import app
        import httpx
        
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/settings")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "sample_rate" in data
                assert "standard_workload_duration" in data
                assert "max_queue_wait" in data
                assert "warning_gpu_temp" in data
                assert "warning_gpu_util" in data
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_update_settings_endpoint(self, temp_db_path):
        """Test POST /api/settings updates settings."""
        import monitor.store
        from monitor.web.routes import app
        import httpx
        
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            # Update settings via API
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/settings",
                    json={
                        "sample_rate": 2,
                        "standard_workload_duration": 20,
                        "max_queue_wait": 60,
                        "warning_gpu_temp": 80.0,
                        "warning_gpu_util": 90.0
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["sample_rate"] == 2
                
                # Verify persistence via GET (same client session)
                response = await client.get("/api/settings")
                assert response.json()["sample_rate"] == 2
        finally:
            monitor.store.DB_PATH = original_path


class TestSettingsIntegration:
    """Integration tests for settings with benchmark runs."""
    
    @pytest.mark.asyncio
    async def test_custom_run_tagged_when_params_modified(self, temp_db_path):
        """Test that a run with modified suite parameters is tagged as custom."""
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner
        from tests.fake_llama_server import FakeLLaMAServer
        
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            async with FakeLLaMAServer(port=18110) as server:
                server.active_requests = 0
                server.set_streaming_config({
                    "first_chunk_delay_ms": 10,
                    "delay_ms": 5,
                    "tokens_per_chunk": 8
                })
                
                runner = BenchmarkRunner()
                
                # Use non-default max_tokens (should be custom with default 10s duration -> 512 tokens)
                run = await runner.trigger_run(max_tokens=1024)
                
                # Run to completion
                from monitor.store import get_benchmark_run
                row = await get_benchmark_run(run.run_id)
                
                assert row["standard_run"] == 0  # custom
                assert "custom" in row["tags"]
        finally:
            monitor.store.DB_PATH = original_path
