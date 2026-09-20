"""Benchmark runner - orchestrates benchmark runs against detected LLM servers."""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx

from .store import (
    init_db, insert_benchmark_run, update_benchmark_run, get_benchmark_run,
    get_loaded_model, insert_model
)
from .probes import ServerDetectorImpl, ModelDetectorImpl, run_probes
from .telemetry.real import RealMemorySource, RealGPUSource, RealGPUMemorySource
from .telemetry.interface import TelemetrySource
from .workloads import LongContextWorkloadRunner, BurstWorkloadRunner, WorkloadResult


class BenchmarkState:
    """States for the benchmark run state machine."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ABORTED = "aborted"


class BenchmarkRun:
    """Represents a benchmark run with its state and metrics."""
    
    def __init__(
        self,
        run_id: int,
        model_name: str,
        server_type: str,
        server_port: int,
        state: str = BenchmarkState.QUEUED,
        ttft: Optional[float] = None,
        peak_gen_tok_s: Optional[float] = None,
        total_time: Optional[float] = None,
        abort_reason: Optional[str] = None
    ):
        self.run_id = run_id
        self.model_name = model_name
        self.server_type = server_type
        self.server_port = server_port
        self.state = state
        self.ttft = ttft
        self.peak_gen_tok_s = peak_gen_tok_s
        self.total_time = total_time
        self.abort_reason = abort_reason
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.run_id,
            "model_name": self.model_name,
            "server_type": self.server_type,
            "server_port": self.server_port,
            "state": self.state,
            "ttft": self.ttft,
            "peak_gen_tok_s": self.peak_gen_tok_s,
            "total_time": self.total_time,
            "abort_reason": self.abort_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class BenchmarkRunner:
    """Runner for benchmark workloads against detected LLM servers.
    
    State machine: queued -> running -> done | aborted(reason)
    One run at a time; subsequent triggers return current run status.
    
    Queue-until-idle: Poll server's active-request gauge every 2s while queued;
    timeout at max_queue_wait seconds with explanatory abort.
    """
    
    def __init__(
        self,
        max_queue_wait: int = 120,  # 2 minutes
        queue_poll_interval: float = 2.0,
        server_detector: Optional[ServerDetectorImpl] = None,
        model_detector: Optional[ModelDetectorImpl] = None
    ):
        self.max_queue_wait = max_queue_wait
        self.queue_poll_interval = queue_poll_interval
        self.server_detector = server_detector or ServerDetectorImpl()
        self.model_detector = model_detector or ModelDetectorImpl()
        
        self._current_run: Optional[BenchmarkRun] = None
        self._lock = asyncio.Lock()
        self._run_task: Optional[asyncio.Task] = None
        
        # Telemetry sources for footprint sampling
        self._memory_source = RealMemorySource()
        self._gpu_source = RealGPUSource()
        self._gpu_memory_source = RealGPUMemorySource()
    
    async def get_status(self) -> Optional[BenchmarkRun]:
        """Get current benchmark run status.
        
        Returns:
            Current BenchmarkRun if one exists, None otherwise
        """
        async with self._lock:
            return self._current_run
    
    async def trigger_run(
        self,
        workload_type: str = "standard",
        max_tokens: int = 512,
        prompt_tokens: int = 64,
        tag: str = "standard"
    ) -> BenchmarkRun:
        """Trigger a new benchmark run.
        
        Acquires the lock exactly once, then spawns the run task.
        If a run is already queued/running, returns that run instead.
        
        The tag is enforced: deviations from suite defaults make the run custom.
        """
        async with self._lock:
            if self._current_run and self._current_run.state in (
                BenchmarkState.QUEUED, BenchmarkState.RUNNING
            ):
                return self._current_run
            
            # Determine if this is a standard or custom run by comparing to defaults
            is_standard = self._is_standard_run(max_tokens, prompt_tokens, tag)
            enforced_tag = "standard" if is_standard else "custom"
            
            run = await self._create_run(
                workload_type=workload_type,
                max_tokens=max_tokens,
                prompt_tokens=prompt_tokens,
                tag=enforced_tag,
            )
            
            workload_params = {
                "workload_type": workload_type,
                "max_tokens": max_tokens,
                "prompt_tokens": prompt_tokens,
                "tag": enforced_tag
            }
            self._run_task = asyncio.create_task(self._run_benchmark(run, workload_params))
            
            return run
    
    def _is_standard_run(self, max_tokens: int, prompt_tokens: int, tag: str) -> bool:
        """Check if the run uses standard suite parameters.
        
        Returns True only if all parameters match defaults AND tag is "standard".
        """
        if tag != "standard":
            return False
        
        # Compare to defaults
        default_short_max = 512
        default_short_prompt = 64
        
        # For now, just check short workload params (other workloads have fixed defaults)
        if max_tokens != default_short_max or prompt_tokens != default_short_prompt:
            return False
        
        return True
    
    async def start_run(
        self,
        workload_type: str = "standard",
        max_tokens: int = 512,
        prompt_tokens: int = 64,
        tag: str = "standard"
    ) -> BenchmarkRun:
        """Create a run in queued state without starting execution.
        
        Prefer trigger_run(): it also starts the run task. This method exists
        for tests and manual control over when execution begins.
        """
        async with self._lock:
            if self._current_run and self._current_run.state in (
                BenchmarkState.QUEUED, BenchmarkState.RUNNING
            ):
                return self._current_run
            
            return await self._create_run(
                workload_type=workload_type,
                max_tokens=max_tokens,
                prompt_tokens=prompt_tokens,
                tag=tag,
            )
    
    async def _create_run(
        self,
        workload_type: str,
        max_tokens: int,
        prompt_tokens: int,
        tag: str,
    ) -> BenchmarkRun:
        """Create the run row and current-run object. Caller must hold self._lock."""
        # Detect server and model
        servers = await self.server_detector.detect()
        llama_servers = [s for s in servers if s.type == "llama-server" and s.port]
        
        if not llama_servers:
            raise RuntimeError("No llama-server detected")
        
        server = llama_servers[0]
        
        # Get model info
        models = await self.model_detector.detect([server])
        if not models:
            raise RuntimeError(f"No model detected for {server.name}")
        
        model = models[0]
        
        # Insert model if not exists (for historical runs)
        model_id = await insert_model(
            name=model.name,
            quant=model.quant,
            context_length=model.context_length,
            server_type=model.server_type or server.type,
            file_size=model.file_size
        )
        
        # Create benchmark run record (initial state: queued)
        workload_params = json.dumps({
            "workload_type": workload_type,
            "max_tokens": max_tokens,
            "prompt_tokens": prompt_tokens,
            "tag": tag
        })
        
        run_id = await insert_benchmark_run(
            model_id=model_id,
            workload_type=workload_type,
            standard_run=(tag == "standard"),
            model_name=model.name,
            model_quant=model.quant,
            context_length=model.context_length,
            server_type=server.type,
            server_port=server.port,
            state=BenchmarkState.QUEUED,
            workload_params=workload_params,
            tags=tag
        )
        
        self._current_run = BenchmarkRun(
            run_id=run_id,
            model_name=model.name,
            server_type=server.type,
            server_port=server.port,
            state=BenchmarkState.QUEUED
        )
        
        return self._current_run
    
    async def _run_benchmark(self, run: BenchmarkRun, workload_params: Dict[str, Any]) -> None:
        """Execute the benchmark run logic (queued -> running -> done/aborted).
        
        This is the core async task that runs the benchmark suite.
        Stores partial results after each workload for progress tracking.
        """
        try:
            # Phase 1: Queue until idle (or timeout)
            run.started_at = time.time()
            await self._queue_until_idle(run)
            
            if run.state == BenchmarkState.ABORTED:
                return
            
            # Phase 2: Run workload suite
            run.state = BenchmarkState.RUNNING
            
            # Update run to running state
            await update_benchmark_run(run_id=run.run_id, state=BenchmarkState.RUNNING)
            
            # Sample footprint before run
            footprint_before = await self._sample_footprint()
            
            # Load suite settings and run workloads
            suite_params = self._get_suite_params(workload_params)
            workload_results = await self._run_suite(
                run.server_port,
                suite_params,
                on_result=self._make_progress_writer(run.run_id),
            )
            
            # Calculate aggregate metrics from all workloads
            total_time = sum(w.total_time for w in workload_results.values())
            
            # Extract TTFT and peak tok/s from workloads
            ttft = workload_results.get("short").ttft if "short" in workload_results else None
            peak_gen_tok_s = 0.0
            prompt_tok_s = 0.0
            aggregate_tok_s = 0.0
            
            for w in workload_results.values():
                if w.workload_name == "short" or w.workload_name == "burst":
                    peak_gen_tok_s = max(peak_gen_tok_s, w.tokens_per_second)
                if w.workload_name == "long-context":
                    prompt_tok_s = w.tokens_per_second
                if w.workload_name == "burst":
                    aggregate_tok_s = w.tokens_per_second
            
            peak_gen_tok_s = peak_gen_tok_s or 0
            prompt_tok_s = prompt_tok_s or 0
            aggregate_tok_s = aggregate_tok_s or 0
            
            # Sample footprint during run (continuously during generation)
            footprint_during = await self._sample_footprint()
            
            # Sample footprint after run
            await asyncio.sleep(1)  # Brief wait for settling
            footprint_after = await self._sample_footprint()
            
            # Store workload results as JSON
            results_json = json.dumps({
                name: result.to_dict() for name, result in workload_results.items()
            })
            
            # Update run with final metrics
            await update_benchmark_run(
                run_id=run.run_id,
                total_time=total_time,
                ttft=ttft,
                prompt_tok_s=prompt_tok_s,
                gen_tok_s=peak_gen_tok_s,
                peak_gen_tok_s=peak_gen_tok_s,
                concurrent_throughput=aggregate_tok_s,
                workload_results=results_json,
                memory_before=json.dumps(footprint_before.get("memory", {})),
                memory_during=json.dumps(footprint_during.get("memory", {})),
                memory_after=json.dumps(footprint_after.get("memory", {})),
                gpu_before=json.dumps(footprint_before.get("gpu", {})),
                gpu_during=json.dumps(footprint_during.get("gpu", {})),
                gpu_after=json.dumps(footprint_after.get("gpu", {}))
            )
            
            run.state = BenchmarkState.DONE
            run.ttft = ttft
            run.peak_gen_tok_s = peak_gen_tok_s
            run.total_time = total_time
            run.finished_at = time.time()
            
            # Final update with results for progress endpoint
            await update_benchmark_run(run_id=run.run_id, workload_results=results_json)
            
        except Exception as e:
            run.state = BenchmarkState.ABORTED
            run.abort_reason = f"Unexpected error: {str(e)}"
            run.finished_at = time.time()
            # Store partial results on abort
            partial_results = {"error": str(e)}
            await update_benchmark_run(
                run_id=run.run_id,
                state=run.state,
                abort_reason=run.abort_reason,
                workload_results=json.dumps(partial_results)
            )
        finally:
            # Persist final state (done or aborted) with the abort reason if any
            await update_benchmark_run(
                run_id=run.run_id,
                state=run.state,
                abort_reason=run.abort_reason,
            )
    
    async def _queue_until_idle(self, run: BenchmarkRun) -> None:
        """Wait until server has no active requests or timeout.
        
        Polls the llamacpp:requests_processing gauge every queue_poll_interval.
        Times out at max_queue_wait seconds.
        """
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                elapsed = time.time() - start_time
                
                # Check timeout
                if elapsed > self.max_queue_wait:
                    run.state = BenchmarkState.ABORTED
                    run.abort_reason = (
                        f"Server busy - active requests did not drain within "
                        f"{self.max_queue_wait}s timeout"
                    )
                    return
                
                try:
                    response = await client.get(
                        f"http://localhost:{run.server_port}/metrics",
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        metrics = response.text
                        
                        # Parse active requests from metrics
                        active_requests = self._parse_active_requests(metrics)
                        
                        if active_requests == 0:
                            # Server is idle
                            return
                    else:
                        # Non-200: keep waiting (server may be restarting); the
                        # timeout check above is the real abort condition
                        pass
                    
                    await asyncio.sleep(self.queue_poll_interval)
                        
                except httpx.RequestError as e:
                    run.state = BenchmarkState.ABORTED
                    run.abort_reason = (
                        f"Server unreachable while queued: {str(e)}"
                    )
                    return
    
    GEN_WINDOW_SECONDS = 0.05  # coalesce buffered SSE bursts into rate windows
    
    async def _run_workload(self, port: int, max_tokens: int) -> tuple:
        """Run the actual benchmark workload against the OpenAI-compatible endpoint.
        
        Sends a small prompt, measures time to first token and peak generation rate.
        Returns (ttft, peak_gen_tok_s).
        """
        # Create a small prompt (we'll estimate token count)
        prompt_text = "Once upon a time in a digital realm, there was a model that could generate"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            request = {
                "model": "benchmark-model",
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": max_tokens,
                "stream": True
            }
            
            ttft = None
            tokens_generated = 0
            gen_rates: List[float] = []
            last_chunk_time = None
            self._window_start = None
            self._window_tokens = 0
            
            request_start_time = time.time()
            
            try:
                async with client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        if line.strip() == "data: [DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line[6:])  # Skip "data: " prefix
                        except json.JSONDecodeError:
                            continue
                        
                        # Get chunk timestamp
                        chunk_time = time.time()
                        
                        if ttft is None and chunk.get("choices"):
                            # First content chunk - capture TTFT
                            ttft = chunk_time - request_start_time
                            last_chunk_time = chunk_time
                        
                        # Track generation rate (rates only; TTFT is a time, not a rate).
                        # Chunks buffered in one TCP segment arrive microseconds apart,
                        # so raw per-chunk deltas measure network batching, not token
                        # speed. Coalesce bursts: a window closes only when a chunk
                        # arrives >= GEN_WINDOW_SECONDS after the window opened.
                        if ttft is not None:
                            tokens_in_chunk = self._count_tokens_in_chunk(chunk)
                            if tokens_in_chunk > 0:
                                if self._window_start is None:
                                    self._window_start = chunk_time
                                    self._window_tokens = 0
                                self._window_tokens += tokens_in_chunk
                                if chunk_time - self._window_start >= self.GEN_WINDOW_SECONDS:
                                    gen_rates.append(self._window_tokens / (chunk_time - self._window_start))
                                    self._window_start = chunk_time
                                    self._window_tokens = 0
                                last_chunk_time = chunk_time
                                tokens_generated += tokens_in_chunk
                        
                    # Flush any open window, then peak generation rate (0 when nothing streamed)
                    if self._window_start is not None and self._window_tokens > 0 and last_chunk_time:
                        window_duration = max(last_chunk_time - self._window_start, self.GEN_WINDOW_SECONDS)
                        gen_rates.append(self._window_tokens / window_duration)
                    self._window_start = None
                    self._window_tokens = 0
                    peak_gen_tok_s = max(gen_rates) if gen_rates else 0
                    
                    return ttft or 0, peak_gen_tok_s, tokens_generated
                    
            except httpx.RequestError as e:
                return 0, 0, tokens_generated
    
    def _parse_active_requests(self, metrics_text: str) -> int:
        """Parse the llamacpp:requests_processing gauge from metrics text."""
        for line in metrics_text.split("\n"):
            if line.startswith("llamacpp:requests_processing"):
                try:
                    return int(float(line.split()[-1]))
                except (ValueError, IndexError):
                    return 0
        return 0
    
    def _count_tokens_in_chunk(self, chunk: Dict[str, Any]) -> int:
        """Count tokens in a chunk from the streaming response."""
        try:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                if content and content != "[DONE]":
                    # Rough token count: 1 token ≈ 4 chars
                    return len(content) // 4
        except (IndexError, KeyError):
            pass
        
        # Check usage field
        usage = chunk.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            return completion_tokens
        
        return 0
    
    def _get_suite_params(self, workload_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get suite parameters, merging workload_params with defaults."""
        defaults = {
            "short": {
                "prompt_tokens": 64,
                "max_tokens": 512,
            },
            "long-context": {
                "prompt_tokens": 16384,
                "max_tokens": 32,
            },
            "burst": {
                "concurrency": 4,
                "tokens_per_request": 128,
            }
        }
        
        # Merge user parameters with defaults
        for name, params in workload_params.items():
            if name in defaults:
                defaults[name].update(params)
        
        return defaults
    
    async def _run_suite(
        self,
        server_port: int,
        suite_params: Dict[str, Dict[str, Any]],
        on_result=None,
    ) -> Dict[str, WorkloadResult]:
        """Run all workloads in the suite and return results.
        
        Args:
            on_result: Optional async callback(name, snapshot) invoked after each
                workload completes, so progress is visible mid-run (the progress
                endpoint reads these partial results from the store).
        """
        results: Dict[str, WorkloadResult] = {}
        
        async def _record(name: str, result: WorkloadResult):
            results[name] = result
            if on_result:
                await on_result(name, dict(results))
        
        # Run short workload (existing implementation)
        short_runner = self._make_short_runner(suite_params.get("short", {}))
        await _record("short", await short_runner.run(server_port))
        
        # Run long-context workload
        long_context_runner = LongContextWorkloadRunner(
            prompt_tokens=suite_params["long-context"]["prompt_tokens"],
            generation_tokens=suite_params["long-context"]["max_tokens"],
            server_port=server_port
        )
        await _record("long-context", await long_context_runner.run(server_port))
        
        # Run burst workload
        burst_runner = BurstWorkloadRunner(
            concurrency=suite_params["burst"]["concurrency"],
            tokens_per_request=suite_params["burst"]["tokens_per_request"],
            server_port=server_port
        )
        await _record("burst", await burst_runner.run(server_port))
        
        return results
    
    def _make_progress_writer(self, run_id: int):
        """Persist partial workload results after each workload completes,
        so /api/benchmarks/progress can name the active workload and ETA mid-run."""
        async def _write(name: str, snapshot: Dict[str, WorkloadResult]):
            await update_benchmark_run(
                run_id=run_id,
                workload_results=json.dumps(
                    {n: r.to_dict() for n, r in snapshot.items()}
                ),
            )
        return _write
    
    def _make_short_runner(self, params: Dict[str, Any]):
        """Create a short workload runner (legacy implementation)."""
        return ShortWorkloadRunner(
            max_tokens=params.get("max_tokens", 512),
            prompt_tokens=params.get("prompt_tokens", 64)
        )
    
    async def _sample_footprint(self) -> Dict[str, Dict[str, Any]]:
        """Sample memory and GPU footprint.
        
        Returns:
            Dict with 'memory' and 'gpu' keys containing sample data
        """
        memory_data = await self._memory_source.collect()
        gpu_data = await self._gpu_source.collect()
        gpu_memory_data = await self._gpu_memory_source.collect()
        
        return {
            "memory": memory_data,
            "gpu": {**gpu_data, "processes": gpu_memory_data.get("gpu_processes", [])}
        }


class ShortWorkloadRunner:
    """Short workload runner - original implementation."""
    
    def __init__(
        self,
        max_tokens: int = 512,
        prompt_tokens: int = 64
    ):
        self.max_tokens = max_tokens
        self.prompt_tokens = prompt_tokens
    
    async def run(self, port: int) -> WorkloadResult:
        """Run the short workload."""
        prompt_text = "Once upon a time in a digital realm, there was a model that could generate"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            request = {
                "model": "benchmark-model",
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": self.max_tokens,
                "stream": True
            }
            
            ttft = None
            tokens_generated = 0
            gen_rates: List[float] = []
            last_chunk_time = None
            window_start = None
            window_tokens = 0
            
            request_start_time = time.time()
            
            try:
                async with client.stream("POST", f"http://localhost:{port}/v1/chat/completions", json=request) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        if line.strip() == "data: [DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        
                        chunk_time = time.time()
                        
                        if ttft is None and chunk.get("choices"):
                            ttft = chunk_time - request_start_time
                            last_chunk_time = chunk_time
                        
                        if ttft is not None:
                            tokens_in_chunk = self._count_tokens_in_chunk(chunk)
                            if tokens_in_chunk > 0:
                                if window_start is None:
                                    window_start = chunk_time
                                    window_tokens = 0
                                window_tokens += tokens_in_chunk
                                if chunk_time - window_start >= 0.05:  # GEN_WINDOW_SECONDS
                                    gen_rates.append(window_tokens / (chunk_time - window_start))
                                    window_start = chunk_time
                                    window_tokens = 0
                                last_chunk_time = chunk_time
                                tokens_generated += tokens_in_chunk
                    
                    # Flush any open window
                    if window_start is not None and window_tokens > 0 and last_chunk_time:
                        window_duration = max(last_chunk_time - window_start, 0.05)
                        gen_rates.append(window_tokens / window_duration)
                    
                    peak_gen_tok_s = max(gen_rates) if gen_rates else 0
                    total_time = time.time() - request_start_time
                    
                    return WorkloadResult(
                        workload_name="short",
                        total_time=total_time,
                        tokens_per_second=peak_gen_tok_s,
                        ttft=ttft,
                        generated_tokens=tokens_generated
                    )
                    
            except httpx.RequestError:
                total_time = time.time() - request_start_time
                return WorkloadResult(
                    workload_name="short",
                    total_time=total_time,
                    tokens_per_second=0,
                    ttft=None,
                    generated_tokens=tokens_generated
                )
    
    def _count_tokens_in_chunk(self, chunk: Dict[str, Any]) -> int:
        """Count tokens in a chunk from the streaming response."""
        try:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                if content and content != "[DONE]":
                    return len(content) // 4
        except (IndexError, KeyError):
            pass
        
        usage = chunk.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            return completion_tokens
        
        return 0
