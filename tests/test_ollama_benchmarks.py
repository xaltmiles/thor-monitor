"""Tests for ollama benchmark support."""

import pytest
import asyncio
import json
import time
import httpx

from monitor.benchmarks import BenchmarkRunner, BenchmarkState, BenchmarkRun
from monitor.workloads import (
    OllamaShortWorkloadRunner,
    OllamaLongContextWorkloadRunner,
    OllamaBurstWorkloadRunner,
    WorkloadResult,
)
from tests.fake_ollama_server import FakeOllamaServer
from monitor.store import DB_PATH, init_db, insert_benchmark_run, get_benchmark_run
from monitor.probes.interface import ServerInfo, ModelInfo


class TestOllamaWorkloadRunners:
    """Tests for ollama workload runners."""
    
    @pytest.mark.asyncio
    async def test_ollama_short_workload_runner_e2e(self):
        """Test short workload runner against real HTTP to fake ollama server."""
        server = FakeOllamaServer(port=11436)
        
        async with server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            runner = OllamaShortWorkloadRunner(
                max_tokens=64,
                prompt_tokens=64,
                model_name="fake-ollama-model"
            )
            
            result = await runner.run(server.port)
            
            assert result.workload_name == "short"
            assert result.total_time > 0
            assert result.tokens_per_second > 0  # Must have real gen tok/s
            assert result.ttft is not None
            assert result.model_name == "fake-ollama-model"
    
    @pytest.mark.asyncio
    async def test_ollama_long_context_workload_runner_e2e(self):
        """Test long-context workload runner against real HTTP to fake ollama server."""
        server = FakeOllamaServer(port=11437)
        
        async with server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            runner = OllamaLongContextWorkloadRunner(
                prompt_tokens=1000,  # Small for testing
                generation_tokens=32,
                model_name="fake-ollama-model"
            )
            
            result = await runner.run(server.port)
            
            assert result.workload_name == "long-context"
            assert result.total_time > 0
            assert result.tokens_per_second > 0  # Must have real prompt tok/s
            assert result.ttft is not None
            assert result.model_name == "fake-ollama-model"
    
    @pytest.mark.asyncio
    async def test_ollama_burst_workload_runner_e2e(self):
        """Test burst workload runner with concurrent requests to fake ollama server."""
        server = FakeOllamaServer(port=11438)
        
        async with server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            runner = OllamaBurstWorkloadRunner(
                concurrency=2,
                tokens_per_request=32,
                model_name="fake-ollama-model"
            )
            
            result = await runner.run(server.port)
            
            assert result.workload_name == "burst"
            assert result.total_time > 0
            assert result.tokens_per_second > 0  # Must have real aggregate tok/s
            assert result.ttft is not None
            assert result.model_name == "fake-ollama-model"
    
    @pytest.mark.asyncio
    async def test_ollama_workload_result_to_dict(self):
        """Test WorkloadResult.to_dict() with model_name."""
        result = WorkloadResult(
            workload_name="short",
            total_time=5.0,
            tokens_per_second=100.0,
            ttft=0.1,
            prompt_tokens=64,
            generated_tokens=512,
            model_name="fake-ollama-model"
        )
        
        data = result.to_dict()
        
        assert data["workload_name"] == "short"
        assert data["total_time"] == 5.0
        assert data["tokens_per_second"] == 100.0
        assert data["ttft"] == 0.1
        assert data["prompt_tokens"] == 64
        assert data["generated_tokens"] == 512
        assert data["model_name"] == "fake-ollama-model"
    
    @pytest.mark.asyncio
    async def test_ollama_short_runner_uses_detected_model(self):
        """Test that ollama short runner uses the detected model name."""
        server = FakeOllamaServer(port=11439)
        server.set_model_name("custom-ollama-model")
        
        async with server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            # Runner with detected model name
            runner = OllamaShortWorkloadRunner(
                max_tokens=64,
                prompt_tokens=64,
                model_name="custom-ollama-model"
            )
            
            result = await runner.run(server.port)
            
            # Should complete successfully with the model
            assert result.workload_name == "short"
            assert result.model_name == "custom-ollama-model"
            assert result.total_time > 0


class TestOllamaBenchmarkIntegration:
    """Integration tests for ollama benchmark suite."""
    
    @pytest.mark.asyncio
    async def test_ollama_benchmark_e2e_with_fake_server(self, temp_db_path):
        """End-to-end test: trigger benchmark and run against fake ollama server."""
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner
        from monitor.probes import ServerDetectorImpl, ModelDetectorImpl
        from monitor.probes.fixtures import FixtureServerDetector, FixtureModelDetector
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeOllamaServer(port=11440) as server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            # Set up fixtures to detect the fake ollama server
            server_detector = FixtureServerDetector([
                ServerInfo(
                    pid=4242,
                    name="ollama",
                    type="ollama",
                    port=server.port,
                    cmdline=["ollama", "serve"],
                )
            ])
            
            # Model detector returns a model with the same name as the server
            model_detector = FixtureModelDetector([
                ModelInfo(
                    name="fake-ollama-model",
                    quant="Q4_K_M",
                    context_length=8192,
                    file_size=4600000000,
                    server_type="ollama",
                )
            ])
            
            await init_db()
            
            runner = BenchmarkRunner(
                max_queue_wait=5,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            
            run = await runner.trigger_run()
            assert run.state == BenchmarkState.QUEUED
            
            # Wait for run to complete
            deadline = asyncio.get_event_loop().time() + 15.0
            while asyncio.get_event_loop().time() < deadline:
                current_run = await runner.get_status()
                if current_run and current_run.state == BenchmarkState.DONE:
                    break
                await asyncio.sleep(0.1)
            
            assert run.state == BenchmarkState.DONE, f"run state: {run.state}"
            assert run.server_type == "ollama"
            assert run.model_name == "fake-ollama-model"
            
            # Verify stored results
            row = await get_benchmark_run(run.run_id)
            assert row["server_type"] == "ollama"
            assert row["model_name"] == "fake-ollama-model"
            
            # Workload results should be stored
            workload_results = json.loads(row.get("workload_results", "{}"))
            assert "short" in workload_results
            assert "long-context" in workload_results
            assert "burst" in workload_results
    
    @pytest.mark.asyncio
    async def test_ollama_benchmark_works_without_detected_model_name(self, temp_db_path):
        """Test that ollama benchmark works even if model name isn't passed."""
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner
        from monitor.probes import ServerDetectorImpl, ModelDetectorImpl
        from monitor.probes.fixtures import FixtureServerDetector, FixtureModelDetector
        from tests.fake_ollama_server import FakeOllamaServer
        from monitor.probes.interface import ServerInfo, ModelInfo
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeOllamaServer(port=11441) as server:
            server.active_requests = 0
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 8
            })
            
            server_detector = FixtureServerDetector([
                ServerInfo(
                    pid=4242,
                    name="ollama",
                    type="ollama",
                    port=server.port,
                    cmdline=["ollama", "serve"],
                )
            ])
            
            model_detector = FixtureModelDetector([
                ModelInfo(
                    name="fallback-model",
                    quant="Q4_K_M",
                    context_length=8192,
                    file_size=4600000000,
                    server_type="ollama",
                )
            ])
            
            await init_db()
            
            runner = BenchmarkRunner(
                max_queue_wait=5,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            
            run = await runner.trigger_run()
            assert run.state == BenchmarkState.QUEUED
            
            deadline = asyncio.get_event_loop().time() + 15.0
            while asyncio.get_event_loop().time() < deadline:
                current_run = await runner.get_status()
                if current_run and current_run.state == BenchmarkState.DONE:
                    break
                await asyncio.sleep(0.1)
            
            assert run.state == BenchmarkState.DONE
            assert run.server_type == "ollama"


class TestOllamaStatsSource:
    """Tests for ollama stats source."""
    
    @pytest.mark.asyncio
    async def test_fixture_ollama_stats_source_namespaced_keys(self):
        """Test that fixture ollama stats use namespaced keys."""
        from monitor.telemetry.fixtures import FixtureOllamaStatsSource
        
        source = FixtureOllamaStatsSource(
            prompt_tokens=1000,
            generated_tokens=500,
            prompt_tokens_rate=10.0,
            generated_tokens_rate=5.0
        )
        
        data = await source.collect()
        
        # Keys should be namespaced
        assert "ollama_prompt_tokens" in data
        assert "ollama_generated_tokens" in data
        assert "ollama_prompt_tokens_rate" in data
        assert "ollama_generated_tokens_rate" in data
        
        # Old keys should NOT be present
        assert "prompt_tokens" not in data
        assert "generated_tokens" not in data
    
    @pytest.mark.asyncio
    async def test_fixture_telemetry_source_namespaces_ollama_stats(self):
        """Test that FixtureTelemetrySource namespaces ollama stats to avoid collision."""
        from monitor.telemetry.fixtures import (
            FixtureTelemetrySource,
            FixtureMemorySource,
            FixtureGPUSource,
            FixtureProcessSource,
            FixtureLLaMAStatsSource,
            FixtureOllamaStatsSource
        )
        
        source = FixtureTelemetrySource(
            memory_source=FixtureMemorySource(total=32_000_000_000, free=16_000_000_000, used=16_000_000_000),
            gpu_source=FixtureGPUSource(util=45.0, temp=70.0, power=150.0),
            process_source=FixtureProcessSource(processes=[]),
            llama_stats_source=FixtureLLaMAStatsSource(
                prompt_tokens=10000,
                generated_tokens=5000,
                prompt_tokens_rate=10.0,
                generated_tokens_rate=5.0
            ),
            ollama_stats_source=FixtureOllamaStatsSource(
                prompt_tokens=2000,
                generated_tokens=1000,
                prompt_tokens_rate=20.0,
                generated_tokens_rate=10.0
            )
        )
        
        data = await source.collect()
        
        # Both llama and ollama stats should be present
        assert "prompt_tokens" in data  # llama
        assert "ollama_prompt_tokens" in data  # ollama
        assert "generated_tokens" in data  # llama
        assert "ollama_generated_tokens" in data  # ollama
        
        # Values should be different
        assert data["prompt_tokens"] == 10000
        assert data["ollama_prompt_tokens"] == 2000
