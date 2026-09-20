"""Tests for probe detection (server and model detection)."""

import pytest
import asyncio
import json
import httpx
from unittest.mock import AsyncMock, patch

from monitor.probes.interface import ServerInfo, ModelInfo
from monitor.probes.server import ServerDetectorImpl
from monitor.probes.model import ModelDetectorImpl
from monitor.probes.probes import run_probes
from monitor.probes.fixtures import FixtureServerDetector, FixtureModelDetector


@pytest.fixture
def sample_server_info():
    """Create sample server info for testing."""
    return ServerInfo(
        pid=1234,
        name="ollama",
        type="ollama",
        port=11434,
        cmdline=["ollama", "serve"]
    )


@pytest.fixture
def sample_model_info():
    """Create sample model info for testing."""
    return ModelInfo(
        name="llama3",
        quant="8b",
        context_length=8192,
        file_size=4600000000,
        server_type="ollama"
    )


class TestServerDetector:
    """Tests for server detection via process scan."""
    
    @pytest.mark.asyncio
    async def test_detect_ollama_server(self):
        """Test detection of ollama server process."""
        detector = ServerDetectorImpl()
        
        # The real detector will find actual processes
        # This test just verifies the method exists and returns a list
        servers = await detector.detect()
        
        assert isinstance(servers, list)
        # May or may not find ollama depending on system state
    
    @pytest.mark.asyncio
    async def test_detect_llama_server(self):
        """Test detection of llama-server process."""
        detector = ServerDetectorImpl()
        
        servers = await detector.detect()
        
        assert isinstance(servers, list)
        # May or may not find llama-server depending on system state
    
    @pytest.mark.asyncio
    async def test_no_false_positives(self):
        """Test that non-server processes are not detected."""
        detector = ServerDetectorImpl()
        
        servers = await detector.detect()
        
        # Verify server type is correctly identified
        for server in servers:
            assert server.type in ("ollama", "llama-server")


class TestModelDetector:
    """Tests for model detection via API."""
    
    @pytest.mark.asyncio
    async def test_no_false_positives_without_servers(self):
        """Test that model detector handles empty server list."""
        detector = ModelDetectorImpl()
        
        models = await detector.detect([])
        
        assert models == []
    
    @pytest.mark.asyncio
    async def test_ollama_detection_fallback(self):
        """Test ollama model detection when API unavailable."""
        detector = ModelDetectorImpl(client_timeout=0.1)
        
        # With no ollama running, should return empty
        servers = [ServerInfo(
            pid=1234,
            name="ollama",
            type="ollama",
            port=11434,
            cmdline=["ollama", "serve"]
        )]
        
        models = await detector.detect(servers)
        
        # Should not crash, but may return empty if API unavailable
        assert isinstance(models, list)
    
    @staticmethod
    def _ollama_ps_response(models: list[dict]) -> httpx.MockTransport:
        """Build a mock transport serving the real ollama GET /api/ps shape."""
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": models})
            return httpx.Response(404)
        return httpx.MockTransport(handler)
    
    @pytest.mark.asyncio
    async def test_ollama_running_model_api(self):
        """Loaded model resolved via ollama's running-model API (GET /api/ps)."""
        transport = self._ollama_ps_response([
            {
                "name": "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                "model": "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                "size": 28665435910,
                "details": {"parameter_size": "34.7B", "quantization_level": "Q4_K_M"},
                "context_length": 262144,
            }
        ])
        detector = ModelDetectorImpl(transport=transport)
        servers = [ServerInfo(pid=1, name="ollama", type="ollama", port=11434, cmdline=["ollama", "serve"])]
        
        models = await detector.detect(servers)
        
        assert len(models) == 1
        m = models[0]
        assert m.name == "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
        assert m.quant == "Q4_K_M"
        assert m.file_size == 28665435910
        assert m.context_length == 262144
        assert m.server_type == "ollama"
    
    @pytest.mark.asyncio
    async def test_ollama_two_models_resident_warns(self):
        """Two models resident in one ollama server produce both entries -> warning."""
        transport = self._ollama_ps_response([
            {"name": "llama3:8b", "size": 4600000000,
             "details": {"quantization_level": "Q4_0"}, "context_length": 8192},
            {"name": "mistral:7b", "size": 4200000000,
             "details": {"quantization_level": "Q4_K_M"}, "context_length": 4096},
        ])
        servers = [ServerInfo(pid=1, name="ollama", type="ollama", port=11434, cmdline=["ollama", "serve"])]
        
        result = await run_probes(
            server_detector=FixtureServerDetector(servers),
            model_detector=ModelDetectorImpl(transport=transport),
        )
        
        assert len(result.models) == 2
        assert result.warning is not None
        assert "llama3:8b" in result.warning
        assert "mistral:7b" in result.warning
    
    @pytest.mark.asyncio
    async def test_ollama_api_error_returns_empty(self):
        """A 404 from the ollama API yields no models and no crash."""
        transport = httpx.MockTransport(lambda req: httpx.Response(404))
        detector = ModelDetectorImpl(transport=transport)
        servers = [ServerInfo(pid=1, name="ollama", type="ollama", port=11434, cmdline=["ollama", "serve"])]
        
        models = await detector.detect(servers)
        
        assert models == []
    
    @pytest.mark.asyncio
    async def test_llama_server_short_flags(self):
        """llama.cpp-style cmdline (-m, -c, --port) is parsed correctly."""
        transport = httpx.MockTransport(lambda req: httpx.Response(404))  # /info unavailable
        detector = ModelDetectorImpl(transport=transport)
        servers = [ServerInfo(
            pid=2, name="llama-server", type="llama-server", port=45771,
            cmdline=[
                "/home/eqr/llama.cpp-gpu/build/bin/llama-server",
                "-m", "/home/eqr/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/x/Qwen3-Coder-Next-Q4_K_M.gguf",
                "--port", "45771",
                "--parallel", "4",
                "-c", "262144",
            ],
        )]
        
        models = await detector.detect(servers)
        
        assert len(models) == 1
        m = models[0]
        assert m.name == "Qwen3-Coder-Next-Q4_K_M"
        assert m.quant == "Q4_K_M"
        assert m.context_length == 262144
        assert m.server_type == "llama-server"
    
    @pytest.mark.asyncio
    async def test_llama_server_alias_and_attached_short_flag(self):
        """--alias overrides the display name; quant falls back to the file stem."""
        transport = httpx.MockTransport(lambda req: httpx.Response(404))
        detector = ModelDetectorImpl(transport=transport)
        servers = [ServerInfo(
            pid=3, name="llama-server", type="llama-server", port=8080,
            cmdline=["llama-server", "-m", "/models/coder-Q4_K_M.gguf", "--alias", "coder", "-c4096"],
        )]
        
        models = await detector.detect(servers)
        
        assert len(models) == 1
        assert models[0].name == "coder"
        assert models[0].quant == "Q4_K_M"  # from filename, despite alias lacking it
        assert models[0].context_length == 4096


class TestProbeIntegration:
    """Integration tests for probe orchestration."""
    
    @pytest.mark.asyncio
    async def test_run_probes_with_fixture(self):
        """Test run_probes with fixture detectors."""
        from monitor.probes import ProbeResult
        
        servers = [
            ServerInfo(
                pid=1234,
                name="ollama",
                type="ollama",
                port=11434,
                cmdline=["ollama", "serve"]
            )
        ]
        
        models = [
            ModelInfo(
                name="llama3",
                quant="8b",
                context_length=8192,
                file_size=4600000000,
                server_type="ollama"
            )
        ]
        
        result = await run_probes(
            server_detector=FixtureServerDetector(servers),
            model_detector=FixtureModelDetector(models)
        )
        
        assert len(result.servers) == 1
        assert result.servers[0].name == "ollama"
        assert len(result.models) == 1
        assert result.models[0].name == "llama3"
        assert result.warning is None
    
    @pytest.mark.asyncio
    async def test_multi_model_warning(self):
        """Test warning when multiple models detected."""
        servers = [
            ServerInfo(pid=1, name="ollama", type="ollama", port=11434, cmdline=["ollama", "serve"]),
            ServerInfo(pid=2, name="llama-server", type="llama-server", port=8080, cmdline=["llama-server"]),
        ]
        
        models = [
            ModelInfo(name="llama3", quant="8b", context_length=8192, file_size=4600000000, server_type="ollama"),
            ModelInfo(name="mistral", quant="Q4_K_M", context_length=8192, file_size=4200000000, server_type="llama-server"),
        ]
        
        result = await run_probes(
            server_detector=FixtureServerDetector(servers),
            model_detector=FixtureModelDetector(models)
        )
        
        assert result.warning is not None
        assert "Multiple loaded models" in result.warning
        assert "llama3" in result.warning
        assert "mistral" in result.warning


class TestFixtureDetectors:
    """Tests for fixture detectors."""
    
    @pytest.mark.asyncio
    async def test_fixture_server_detector(self):
        """Test FixtureServerDetector returns configured servers."""
        expected_servers = [
            ServerInfo(pid=1234, name="ollama", type="ollama", port=11434, cmdline=["ollama", "serve"])
        ]
        
        detector = FixtureServerDetector(expected_servers)
        servers = await detector.detect()
        
        assert len(servers) == 1
        assert servers[0].name == "ollama"
    
    @pytest.mark.asyncio
    async def test_fixture_model_detector(self):
        """Test FixtureModelDetector returns configured models."""
        expected_models = [
            ModelInfo(name="llama3", quant="8b", context_length=8192, file_size=4600000000, server_type="ollama")
        ]
        
        detector = FixtureModelDetector(expected_models)
        models = await detector.detect([])
        
        assert len(models) == 1
        assert models[0].name == "llama3"
