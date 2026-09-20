"""Tests for benchmark functionality."""

import pytest
import asyncio
import json
import time
import httpx
import aiosqlite
from pathlib import Path

from monitor.benchmarks import BenchmarkRunner, BenchmarkState, BenchmarkRun
from tests.fake_llama_server import FakeLLaMAServer
from monitor.store import DB_PATH, init_db, insert_benchmark_run, get_benchmark_run, update_benchmark_run


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""
    
    @pytest.mark.asyncio
    async def test_runner_initializes_without_server(self):
        """BenchmarkRunner can be instantiated without active server."""
        runner = BenchmarkRunner()
        
        status = await runner.get_status()
        assert status is None


class TestBenchmarkStore:
    """Tests for benchmark store functions."""
    
    @pytest.mark.asyncio
    async def test_insert_and_get_benchmark_run(self, temp_db_path):
        """Test inserting and retrieving a benchmark run."""
        from monitor.store import DB_PATH as original_path
        import monitor.store
        monitor.store.DB_PATH = temp_db_path
        
        try:
            await init_db()
            
            # Insert a benchmark run
            run_id = await insert_benchmark_run(
                model_id=1,
                workload_type="standard",
                standard_run=True,
                model_name="test-model",
                server_type="llama-server",
                tags="standard"
            )
            
            assert run_id is not None
            
            # Retrieve it
            run = await get_benchmark_run(run_id)
            
            assert run is not None
            assert run["model_name"] == "test-model"
            assert run["server_type"] == "llama-server"
            assert run["workload_type"] == "standard"
            assert run["tags"] == "standard"
            
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_update_benchmark_run(self, temp_db_path):
        """Test updating a benchmark run with metrics."""
        from monitor.store import DB_PATH as original_path
        import monitor.store
        monitor.store.DB_PATH = temp_db_path
        
        try:
            await init_db()
            
            # Insert a benchmark run
            run_id = await insert_benchmark_run(
                model_id=1,
                workload_type="standard",
                standard_run=True,
                model_name="test-model",
                server_type="llama-server"
            )
            
            # Update with metrics
            await update_benchmark_run(
                run_id=run_id,
                total_time=10.5,
                ttft=0.25,
                gen_tok_s=50.0,
                peak_gen_tok_s=75.0
            )
            
            # Retrieve and verify
            run = await get_benchmark_run(run_id)
            
            assert run["total_time"] == 10.5
            assert run["ttft"] == 0.25
            assert run["gen_tok_s"] == 50.0
            assert run["peak_gen_tok_s"] == 75.0
            
        finally:
            monitor.store.DB_PATH = original_path
    
    @pytest.mark.asyncio
    async def test_store_with_footprint_samples(self, temp_db_path):
        """Test benchmark run with memory and GPU footprint samples."""
        from monitor.store import DB_PATH as original_path
        import monitor.store
        monitor.store.DB_PATH = temp_db_path
        
        try:
            await init_db()
            
            # Footprint samples as JSON
            memory_samples = json.dumps({
                "memory_total": 32_000_000_000,
                "memory_free": 16_000_000_000,
                "memory_used": 16_000_000_000
            })
            
            gpu_samples = json.dumps({
                "gpu_util": 45.0,
                "gpu_temp": 70.0,
                "gpu_power": 150.0,
                "processes": []
            })
            
            # Insert with footprint
            run_id = await insert_benchmark_run(
                model_id=1,
                workload_type="standard",
                standard_run=True,
                model_name="test-model",
                server_type="llama-server",
                memory_before=memory_samples,
                memory_during=memory_samples,
                memory_after=memory_samples,
                gpu_before=gpu_samples,
                gpu_during=gpu_samples,
                gpu_after=gpu_samples
            )
            
            run = await get_benchmark_run(run_id)
            
            assert run["memory_before"] == memory_samples
            assert run["gpu_before"] == gpu_samples
            
        finally:
            monitor.store.DB_PATH = original_path


class TestFakeLLaMAServerStreaming:
    """Tests for fake LLaMA server streaming endpoint."""
    
    @pytest.mark.asyncio
    async def test_streaming_endpoint_basic(self):
        """Test that streaming /v1/chat/completions works."""
        server = FakeLLaMAServer(port=18086)
        
        async with server:
            # Set active requests to 0 for immediate execution
            server.active_requests = 0
            
            async with httpx.AsyncClient() as client:
                request = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 32,
                    "stream": True
                }
                
                async with client.stream("POST", f"{server.url}/v1/chat/completions", json=request) as response:
                    assert response.status_code == 200
                    
                    lines = []
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            lines.append(line)
                    
                    # Should get multiple chunks plus [DONE]
                    assert len(lines) >= 2
    
    @pytest.mark.asyncio
    async def test_streaming_with_configurable_delay(self):
        """Test that streaming delay can be configured for testing."""
        server = FakeLLaMAServer(port=18087)
        
        async with server:
            # Configure for fast testing
            server.set_streaming_config({
                "first_chunk_delay_ms": 10,
                "delay_ms": 5,
                "tokens_per_chunk": 4
            })
            
            # Set active requests to 0
            server.active_requests = 0
            
            async with httpx.AsyncClient() as client:
                request = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 16,
                    "stream": True
                }
                
                start = time.time()
                async with client.stream("POST", f"{server.url}/v1/chat/completions", json=request) as response:
                    async for _ in response.aiter_lines():
                        pass
                elapsed = time.time() - start
                
                # Should complete in ~30ms (10ms TTFT + 3 chunks * 5ms)
                assert elapsed < 1.0  # Much faster than real-time


class TestBenchmarkIntegration:
    """Integration tests with fake LLaMA server over real HTTP."""
    
    @staticmethod
    def _fixture_detectors(port: int):
        """Detectors that 'find' a llama-server on the fake server's port."""
        from monitor.probes import ServerDetectorImpl, ModelDetectorImpl
        from monitor.probes.fixtures import FixtureServerDetector
        from monitor.probes.interface import ServerInfo
        
        server_detector = FixtureServerDetector([
            ServerInfo(
                pid=4242,
                name="llama-server",
                type="llama-server",
                port=port,
                cmdline=["llama-server", "-m", "/models/fake-Q4_K_M.gguf", "--port", str(port)],
            )
        ])
        # /info unavailable on the fake server -> falls back to cmdline parsing
        model_detector = ModelDetectorImpl(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
        return server_detector, model_detector
    
    @staticmethod
    async def _wait_for_state(runner, states, timeout=15.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            run = await runner.get_status()
            if run and run.state in states:
                return run
            await asyncio.sleep(0.1)
        raise TimeoutError(f"run never reached {states}; last={run.state if run else None}")
    
    @pytest.mark.asyncio
    async def test_benchmark_e2e_trigger_to_stored_row(self, temp_db_path):
        """Real end-to-end: trigger -> queue -> run -> stored result, via HTTP-shaped path.
        
        Exercises trigger_run() (the API's code path, including its locking),
        the streaming workload against the fake server, and the stored row.
        """
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner, BenchmarkState
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeLLaMAServer(port=18088) as server:
            await init_db()
            server.active_requests = 0
            server.set_streaming_config({"first_chunk_delay_ms": 20, "delay_ms": 5, "tokens_per_chunk": 8})
            
            server_detector, model_detector = self._fixture_detectors(server.port)
            runner = BenchmarkRunner(
                max_queue_wait=5,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            
            run = await runner.trigger_run(max_tokens=64)
            assert run.state == BenchmarkState.QUEUED
            
            run = await self._wait_for_state(runner, (BenchmarkState.DONE, BenchmarkState.ABORTED))
            assert run.state == BenchmarkState.DONE, f"run aborted: {run.abort_reason}"
            assert run.ttft is not None and run.ttft > 0
            assert run.peak_gen_tok_s is not None and run.peak_gen_tok_s > 0
            assert run.total_time is not None and run.total_time > 0
            
            # Stored row carries model identity, server type/port, workload params, tag
            row = await get_benchmark_run(run.run_id)
            assert row["state"] == "done"
            assert row["model_name"] == "fake-Q4_K_M"
            assert row["model_quant"] == "Q4_K_M"
            assert row["server_type"] == "llama-server"
            assert row["server_port"] == server.port
            assert row["ttft"] == run.ttft
            assert row["peak_gen_tok_s"] == run.peak_gen_tok_s
            # max_tokens=64 is a deviation from the 512 default, so tagged as custom
            assert "custom" in row["tags"]
            params = json.loads(row["workload_params"])
            assert params["max_tokens"] == 64
            assert row["standard_run"] == 0  # custom run
            
            # Footprint sampled before/during/after and stored
            for col in ("memory_before", "memory_during", "memory_after", "gpu_before", "gpu_during", "gpu_after"):
                assert row[col], f"{col} not stored"
            assert "memory_total" in json.loads(row["memory_before"])
    
    @pytest.mark.asyncio
    async def test_benchmark_queues_until_server_drains(self, temp_db_path):
        """Run waits while the server reports active requests, then proceeds."""
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner, BenchmarkState
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeLLaMAServer(port=18086) as server:
            await init_db()
            server.active_requests = 1  # busy
            server.set_streaming_config({"first_chunk_delay_ms": 10, "delay_ms": 5, "tokens_per_chunk": 8})
            
            server_detector, model_detector = self._fixture_detectors(server.port)
            runner = BenchmarkRunner(
                max_queue_wait=10,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            
            run = await runner.trigger_run(max_tokens=32)
            assert run.state == BenchmarkState.QUEUED
            
            # Still queued while the server is busy
            await asyncio.sleep(0.7)
            run = await runner.get_status()
            assert run.state == BenchmarkState.QUEUED
            
            # Drain: the run should now proceed to done
            server.active_requests = 0
            run = await self._wait_for_state(runner, (BenchmarkState.DONE, BenchmarkState.ABORTED))
            assert run.state == BenchmarkState.DONE, f"run aborted: {run.abort_reason}"
    
    @pytest.mark.asyncio
    async def test_benchmark_queue_timeout_aborts_with_reason(self, temp_db_path):
        """Server stays busy -> run aborts at the queue timeout with a stored reason."""
        import monitor.store
        from monitor.benchmarks import BenchmarkRunner, BenchmarkState
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeLLaMAServer(port=18087) as server:
            await init_db()
            server.active_requests = 3  # stays busy the whole time
            
            server_detector, model_detector = self._fixture_detectors(server.port)
            runner = BenchmarkRunner(
                max_queue_wait=1,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            
            run = await runner.trigger_run(max_tokens=32)
            run = await self._wait_for_state(runner, (BenchmarkState.ABORTED,), timeout=10.0)
            
            assert run.state == BenchmarkState.ABORTED
            assert run.abort_reason and "drain" in run.abort_reason
            
            row = await get_benchmark_run(run.run_id)
            assert row["state"] == "aborted"
            assert row["abort_reason"] == run.abort_reason
    
    @pytest.mark.asyncio
    async def test_api_uses_single_shared_runner(self, temp_db_path):
        """The API layer shares one runner: concurrent triggers return the same run.
        
        Guards against per-request BenchmarkRunner instantiation (state would
        never be shared) and against the trigger lock self-deadlocking.
        """
        import monitor.store
        from monitor.web.routes import app as routes_app
        from monitor.benchmarks import BenchmarkRunner
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        await init_db()
        
        async with FakeLLaMAServer(port=18089) as server:
            server.active_requests = 0
            server.set_streaming_config({"first_chunk_delay_ms": 50, "delay_ms": 20, "tokens_per_chunk": 8})
            
            server_detector, model_detector = self._fixture_detectors(server.port)
            test_runner = BenchmarkRunner(
                max_queue_wait=5,
                queue_poll_interval=0.2,
                server_detector=server_detector,
                model_detector=model_detector,
            )
            routes_app.state.benchmark_runner = test_runner
            
            transport = httpx.ASGITransport(app=routes_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Two overlapping triggers -> same run, no deadlock
                r1, r2 = await asyncio.gather(
                    client.post("/api/benchmarks/start", timeout=15),
                    client.post("/api/benchmarks/start", timeout=15),
                )
                assert r1.status_code == 200 and r2.status_code == 200
                id1 = r1.json()["run"]["id"]
                id2 = r2.json()["run"]["id"]
                assert id1 == id2
                
                # Status endpoint sees the same shared run
                r3 = await client.get("/api/benchmarks/status", timeout=10)
                assert r3.status_code == 200
                body = r3.json()
                assert body["run"] is not None
                assert body["run"]["id"] == id1
                assert body["status"] in ("queued", "running", "done")


class TestBenchmarkStateMachine:
    """Tests for benchmark state machine logic."""
    
    @pytest.mark.asyncio
    async def test_benchmark_state_enum(self):
        """Test BenchmarkState constants."""
        assert BenchmarkState.QUEUED == "queued"
        assert BenchmarkState.RUNNING == "running"
        assert BenchmarkState.DONE == "done"
        assert BenchmarkState.ABORTED == "aborted"
    
    @pytest.mark.asyncio
    async def test_benchmark_run_to_dict(self):
        """Test BenchmarkRun.to_dict() serialization."""
        run = BenchmarkRun(
            run_id=1,
            model_name="test-model",
            server_type="llama-server",
            server_port=8080,
            state=BenchmarkState.QUEUED
        )
        
        run_dict = run.to_dict()
        
        assert run_dict["id"] == 1
        assert run_dict["model_name"] == "test-model"
        assert run_dict["state"] == BenchmarkState.QUEUED
        assert "started_at" in run_dict
        assert "finished_at" in run_dict


class TestSuiteWorkloads:
    """Tests for suite workload runners (long-context and burst)."""
    
    @pytest.mark.asyncio
    async def test_long_context_workload_runner_e2e(self, temp_db_path):
        """Test long-context workload runner against real HTTP to fake server."""
        import monitor.store
        from monitor.workloads import LongContextWorkloadRunner, WorkloadResult
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeLLaMAServer(port=18100) as server:
            await init_db()
            server.active_requests = 0
            server.set_streaming_config({"first_chunk_delay_ms": 10, "delay_ms": 5, "tokens_per_chunk": 8})
            
            runner = LongContextWorkloadRunner(
                prompt_tokens=1000,  # Small for testing
                generation_tokens=32
            )
            
            result = await runner.run(server.port)
            
            assert result.workload_name == "long-context"
            assert result.total_time > 0
            assert result.tokens_per_second > 0  # Must have real prompt tok/s
            assert result.generated_tokens > 0  # Server generates tokens during generation phase
    
    @pytest.mark.asyncio
    async def test_burst_workload_runner_e2e(self, temp_db_path):
        """Test burst workload runner with concurrent requests to fake server."""
        import monitor.store
        from monitor.workloads import BurstWorkloadRunner, WorkloadResult
        from tests.fake_llama_server import FakeLLaMAServer
        
        monitor.store.DB_PATH = temp_db_path
        
        async with FakeLLaMAServer(port=18101) as server:
            await init_db()
            server.active_requests = 0
            server.set_streaming_config({"first_chunk_delay_ms": 10, "delay_ms": 5, "tokens_per_chunk": 8})
            
            runner = BurstWorkloadRunner(
                concurrency=2,
                tokens_per_request=32
            )
            
            result = await runner.run(server.port)
            
            assert result.workload_name == "burst"
            assert result.total_time > 0
            assert result.tokens_per_second > 0  # Must have real aggregate tok/s
            assert result.generated_tokens > 0
    
    @pytest.mark.asyncio
    async def test_workload_result_to_dict(self):
        """Test WorkloadResult.to_dict() serialization."""
        from monitor.workloads import WorkloadResult
        
        result = WorkloadResult(
            workload_name="short",
            total_time=5.0,
            tokens_per_second=100.0,
            ttft=0.1,
            prompt_tokens=64,
            generated_tokens=512
        )
        
        data = result.to_dict()
        
        assert data["workload_name"] == "short"
        assert data["total_time"] == 5.0
        assert data["tokens_per_second"] == 100.0
        assert data["ttft"] == 0.1
        assert data["prompt_tokens"] == 64
        assert data["generated_tokens"] == 512


class TestSuiteSettings:
    """Tests for suite settings and standard vs custom tagging."""
    
    @pytest.mark.asyncio
    async def test_suite_parameters_default(self, temp_db_path):
        """Test default suite parameters."""
        import monitor.store
        original_path = monitor.store.DB_PATH
        monitor.store.DB_PATH = temp_db_path
        try:
            await init_db()
            
            from monitor.benchmarks import BenchmarkRunner
            
            runner = BenchmarkRunner()
            
            params = await runner._get_suite_params({})
            
            assert params["short"]["max_tokens"] == 512
            assert params["long-context"]["prompt_tokens"] == 16384
            assert params["burst"]["concurrency"] == 4
        finally:
            monitor.store.DB_PATH = original_path



    @pytest.mark.asyncio
    async def test_suite_reports_progress_incrementally(self, temp_db_path):
        """Partial results must be persisted after EACH workload, not only at the end.

        Regression guard for the reviewer BLOCKER: /api/benchmarks/progress
        reads workload_results from the store mid-run; if the suite only wrote
        at the end, progress would stay 0/None while running.
        """
        import json as _json

        import monitor.store
        from monitor.benchmarks import BenchmarkRunner
        from monitor.store import get_benchmark_run, insert_benchmark_run, update_benchmark_run
        from tests.fake_llama_server import FakeLLaMAServer

        monitor.store.DB_PATH = temp_db_path

        async with FakeLLaMAServer(port=18101) as server:
            await init_db()
            server.active_requests = 0
            server.set_streaming_config({"first_chunk_delay_ms": 10, "delay_ms": 5, "tokens_per_chunk": 8})

            runner = BenchmarkRunner()

            async with aiosqlite.connect(str(temp_db_path)) as db:
                await db.execute(
                    """INSERT INTO benchmark_runs (model_id, workload_type, standard_run, created_at, state)
                       VALUES (1, 'standard', 1, '2026-01-01T00:00:00+00:00', 'running')"""
                )
                await db.commit()
                run_id_row = await db.execute("SELECT id FROM benchmark_runs ORDER BY id DESC LIMIT 1")
                run_id = (await run_id_row.fetchone())[0]

            snapshots = []

            async def on_result(name, snapshot):
                snapshots.append((name, list(snapshot.keys())))
                # Mirror the production progress writer: persist the snapshot
                await update_benchmark_run(
                    run_id=run_id,
                    workload_results=_json.dumps({n: r.to_dict() for n, r in snapshot.items()}),
                )
                # Mid-run: the store must already hold the completed prefix
                row = await get_benchmark_run(run_id)
                stored = _json.loads(row["workload_results"])
                assert name in stored, f"workload {name} not visible in store immediately after completion"

            suite_params = await runner._get_suite_params({"max_tokens": 32})
            results = await runner._run_suite(server.port, suite_params, on_result=on_result)

            # All three workloads ran, callback fired once per workload in order
            assert list(results.keys()) == ["short", "long-context", "burst"]
            assert [n for n, _ in snapshots] == ["short", "long-context", "burst"]
            assert [list(k) for _, k in snapshots] == [
                ["short"],
                ["short", "long-context"],
                ["short", "long-context", "burst"],
            ]

            row = await get_benchmark_run(run_id)
            stored = _json.loads(row["workload_results"])
            assert set(stored.keys()) == {"short", "long-context", "burst"}
