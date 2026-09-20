"""Test fixtures for monitor tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from monitor.store import DB_PATH, init_db, insert_telemetry_sample, get_latest_telemetry


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_db = temp_dir / "test_monitor.db"
    
    yield temp_db
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_db(temp_db_path):
    """Initialize database with test path."""
    import monitor.store
    original_path = monitor.store.DB_PATH
    monitor.store.DB_PATH = temp_db_path
    
    yield temp_db_path
    
    monitor.store.DB_PATH = original_path


@pytest.fixture
async def populated_db(test_db):
    """Create a database with sample telemetry data."""
    await init_db()
    
    # Insert some test data
    for i in range(10):
        await insert_telemetry_sample(
            memory_total=32_000_000_000,
            memory_free=16_000_000_000 - i * 1_000_000_000,
            memory_used=16_000_000_000 + i * 1_000_000_000,
            gpu_util=45.0 + i,
            gpu_temp=70.0 + i,
            gpu_power=150.0 + i,
            process_memory="[]"
        )
    
    return test_db


@pytest.fixture
def fixture_memory_source():
    """Create a fixture memory source."""
    from monitor.telemetry.fixtures import FixtureMemorySource
    return FixtureMemorySource(
        total=32_000_000_000,
        free=16_000_000_000,
        used=16_000_000_000
    )


@pytest.fixture
def fixture_gpu_source():
    """Create a fixture GPU source."""
    from monitor.telemetry.fixtures import FixtureGPUSource
    return FixtureGPUSource(
        util=45.0,
        temp=70.0,
        power=150.0
    )


@pytest.fixture
def fixture_process_source():
    """Create a fixture process source."""
    from monitor.telemetry.fixtures import FixtureProcessSource
    return FixtureProcessSource(
        processes=[
            {"pid": 1234, "name": "ollama", "rss": 8_000_000_000, "vms": 10_000_000_000, "cmdline": ["ollama", "serve"]},
            {"pid": 5678, "name": "llama-server", "rss": 12_000_000_000, "vms": 15_000_000_000, "cmdline": ["llama-server", "--model", "model.gguf"]},
        ]
    )


@pytest.fixture
def fixture_server_info_ollama():
    """Create fixture ollama server info."""
    from monitor.probes.interface import ServerInfo
    return ServerInfo(
        pid=1234,
        name="ollama",
        type="ollama",
        port=11434,
        cmdline=["ollama", "serve"]
    )


@pytest.fixture
def fixture_server_info_llama():
    """Create fixture llama-server info."""
    from monitor.probes.interface import ServerInfo
    return ServerInfo(
        pid=5678,
        name="llama-server",
        type="llama-server",
        port=8080,
        cmdline=["llama-server", "--model", "/path/to/model.gguf"]
    )


@pytest.fixture
def fixture_model_info():
    """Create fixture model info."""
    from monitor.probes.interface import ModelInfo
    return ModelInfo(
        name="llama3",
        quant="8b",
        context_length=8192,
        file_size=4600000000,
        server_type="ollama"
    )
