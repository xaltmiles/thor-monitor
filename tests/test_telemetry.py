"""Tests for telemetry sources."""

import pytest
import asyncio

from monitor.telemetry.fixtures import (
    FixtureMemorySource, FixtureGPUSource, FixtureProcessSource, 
    FixtureTelemetrySource
)


@pytest.mark.asyncio
async def test_fixture_memory_source():
    """Test fixture memory source."""
    source = FixtureMemorySource(
        total=32_000_000_000,
        free=16_000_000_000,
        used=16_000_000_000
    )
    
    data = await source.collect()
    
    assert data["memory_total"] == 32_000_000_000
    assert data["memory_free"] == 16_000_000_000
    assert data["memory_used"] == 16_000_000_000


@pytest.mark.asyncio
async def test_fixture_gpu_source():
    """Test fixture GPU source."""
    source = FixtureGPUSource(
        util=45.0,
        temp=70.0,
        power=150.0
    )
    
    data = await source.collect()
    
    assert data["gpu_util"] == 45.0
    assert data["gpu_temp"] == 70.0
    assert data["gpu_power"] == 150.0


@pytest.mark.asyncio
async def test_fixture_process_source():
    """Test fixture process source."""
    processes = [
        {"pid": 1234, "name": "ollama", "rss": 8_000_000_000},
        {"pid": 5678, "name": "llama-server", "rss": 12_000_000_000},
    ]
    
    source = FixtureProcessSource(processes=processes)
    
    data = await source.collect()
    
    assert "processes" in data
    assert len(data["processes"]) == 2
    assert data["processes"][0]["name"] == "ollama"
    assert data["processes"][1]["name"] == "llama-server"


@pytest.mark.asyncio
async def test_fixture_telemetry_source():
    """Test combined fixture telemetry source."""
    memory_source = FixtureMemorySource(
        total=32_000_000_000,
        free=16_000_000_000,
        used=16_000_000_000
    )
    
    gpu_source = FixtureGPUSource(
        util=45.0,
        temp=70.0,
        power=150.0
    )
    
    process_source = FixtureProcessSource(processes=[])
    
    source = FixtureTelemetrySource(
        memory_source=memory_source,
        gpu_source=gpu_source,
        process_source=process_source
    )
    
    data = await source.collect()
    
    # Check all data is combined
    assert "memory_total" in data
    assert "memory_free" in data
    assert "memory_used" in data
    assert "gpu_util" in data
    assert "gpu_temp" in data
    assert "gpu_power" in data
    assert "processes" in data
    
    assert data["memory_total"] == 32_000_000_000
    assert data["gpu_util"] == 45.0
