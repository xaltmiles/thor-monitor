"""Real telemetry sources - collect from actual system."""

import asyncio
import re
from collections import deque
from typing import Any, Tuple, Dict
import psutil
import httpx
from .interface import MemorySource, GPUSource, ProcessSource, GPUMemorySource, LLaMAStatsSource, OllamaStatsSource
from ..probes.server import ServerDetectorImpl


def _compute_window_averages(window_samples: list, current_time: float, counter_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Compute rolling window averages for token rates.
    
    Returns dict with *{key}_rate_avg keys computed as counter delta across
    the window divided by elapsed time.
    
    Args:
        window_samples: List of (timestamp, counters_dict) tuples
        current_time: Current monotonic timestamp
        counter_keys: Tuple of key names to average (e.g., ("prompt_tokens", "generated_tokens"))
    """
    result: Dict[str, float] = {}
    
    if len(window_samples) < 2:
        # Not enough samples for averaging
        for key in counter_keys:
            result[f"{key}_rate_avg"] = 0.0
        return result
    
    # Get first and last samples in window
    first_time, first_counters = window_samples[0]
    last_time, last_counters = window_samples[-1]
    
    elapsed = last_time - first_time
    if elapsed <= 0:
        # Same timestamp, no rate can be computed
        for key in counter_keys:
            result[f"{key}_rate_avg"] = 0.0
        return result
    
    # Compute delta across the window
    for key in counter_keys:
        delta = last_counters.get(key, 0) - first_counters.get(key, 0)
        result[f"{key}_rate_avg"] = delta / elapsed
    
    return result


class RealMemorySource(MemorySource):
    """Collect unified memory statistics from /proc/meminfo."""
    
    async def collect(self) -> dict:
        """Collect memory stats from /proc/meminfo."""
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1]) * 1024  # Convert KB to bytes
                    meminfo[key] = value
        
        # Unified memory totals (total available system memory)
        memory_total = meminfo.get("MemTotal", 0)
        memory_free = meminfo.get("MemAvailable", 0)
        memory_used = memory_total - memory_free
        
        return {
            "memory_total": memory_total,
            "memory_free": memory_free,
            "memory_used": memory_used
        }


class RealGPUSource(GPUSource):
    """Collect GPU statistics from nvidia-smi."""
    
    async def collect(self) -> dict:
        """Collect GPU stats using nvidia-smi."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                parts = stdout.decode().strip().split(",")
                if len(parts) >= 3:
                    return {
                        "gpu_util": float(parts[0].strip()),
                        "gpu_temp": float(parts[1].strip()),
                        "gpu_power": float(parts[2].strip().split()[0])  # Remove ' W'
                    }
        except Exception:
            pass
        
        return {
            "gpu_util": None,
            "gpu_temp": None,
            "gpu_power": None
        }


class RealProcessSource(ProcessSource):
    """Collect process table information."""
    
    async def collect(self) -> dict:
        """Collect process table data."""
        processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
            try:
                info = proc.info
                mem_info = info.get("memory_info")
                processes.append({
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "rss": mem_info.rss if mem_info else 0,
                    "vms": mem_info.vms if mem_info else 0,
                    "cmdline": info.get("cmdline")
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {"processes": processes}


class RealGPUMemorySource(GPUMemorySource):
    """Collect per-process GPU memory attribution from nvidia-smi."""
    
    async def collect(self) -> dict:
        """Collect per-process GPU memory using nvidia-smi --query-compute-apps."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                lines = stdout.decode().strip().split("\n")
                processes = []
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        try:
                            processes.append({
                                "pid": int(parts[0].strip()),
                                "name": parts[1].strip(),
                                # nvidia-smi nounits reports MiB; store bytes for
                                # consistency with the rest of the telemetry
                                "gpu_memory": int(parts[2].strip()) * 1024 * 1024
                            })
                        except (ValueError, IndexError):
                            continue
                return {"gpu_processes": processes}
        except Exception:
            pass
        
        return {"gpu_processes": []}


class RealLLaMAStatsSource(LLaMAStatsSource):
    """Collect llama-server metrics from its Prometheus /metrics endpoint.
    
    The llama-server is located via the probe process scan (its port is a
    launch flag, unknowable from config), with a short cache to avoid
    re-scanning the process table on every 1 Hz tick.
    
    ## Throughput measurement methodology
    
    This source measures **aggregate server throughput** across all parallel slots.
    When llama-server is started with `--parallel N`, all slots share the GPU,
    and the counter `llamacpp:tokens_predicted_total` accumulates tokens from
    ALL slots. The rate computed as `delta / elapsed` is therefore:
    
        aggregate_tok_s = sum(all_slot_throughputs)
    
    For comparison:
    - Unsloth Studio typically shows **per-slot or per-request** tok/s
    - This dashboard shows **aggregate server-wide** tok/s
    
    Example: With `--parallel 4` and each slot achieving ~20 tok/s:
    - Dashboard (aggregate): ~80 tok/s
    - Studio (per-slot): ~20 tok/s
    
    Both are "correct" for their respective questions:
    - "How fast is my model per request?" → per-slot (Studio)
    - "How much total throughput is the server producing?" → aggregate (Dashboard)
    
    See also: README section "Understanding tok/s: aggregate vs per-request throughput"
    """
    
    DETECT_CACHE_SECONDS = 10.0
    
    def __init__(self, host: str = None, port: int = None, timeout: float = 2.0, window_seconds: float = 10.0):
        self.host = host  # None -> discovered from probes
        self.port = port  # None -> discovered from probes
        self.timeout = timeout
        self._last_counters = None
        self._last_timestamp = None  # monotonic timestamp of last collect
        self._detect_cache = None  # (monotonic_time, host, port)
        self._window_seconds = window_seconds
        # deque of (monotonic_ts, counters_dict) to track rolling window
        # ~1 sample/sec * window_seconds (window_seconds of 10 = 10 samples)
        self._counter_window = deque(maxlen=int(window_seconds))
    
    async def _resolve_target(self) -> tuple:
        """Find the llama-server to poll: explicit override, else probe scan."""
        if self.host and self.port:
            return self.host, self.port
        
        now = asyncio.get_event_loop().time()
        if self._detect_cache and now - self._detect_cache[0] < self.DETECT_CACHE_SECONDS:
            return self._detect_cache[1], self._detect_cache[2]
        
        host, port = None, None
        try:
            servers = await ServerDetectorImpl().detect()
            llama = next((s for s in servers if s.type == "llama-server" and s.port), None)
            if llama:
                host, port = "localhost", llama.port
        except Exception:
            pass
        
        self._detect_cache = (now, host, port)
        return host, port
    
    async def collect(self) -> dict:
        """Collect llama-server metrics from /metrics endpoint.
        
        Returns:
            dict with cumulative counters (prompt_tokens, generated_tokens,
            speculative_accepts), per-second rates (*_rate) computed by
            differencing against the previous sample, normalized to per-second,
            and averaged rates (*_rate_avg) computed over the rolling window.
        """
        host, port = await self._resolve_target()
        if not host or not port:
            self._last_counters = None
            self._last_timestamp = None
            return self._empty()
        
        current_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"http://{host}:{port}/metrics")
                
                if response.status_code == 200:
                    counters = self._parse_metrics(response.text)
                    
                    result = counters.copy()
                    
                    # Track counter window for rolling average
                    self._counter_window.append((current_time, counters.copy()))
                    
                    if self._last_counters and self._last_timestamp is not None:
                        elapsed = current_time - self._last_timestamp
                        for key in ("prompt_tokens", "generated_tokens", "speculative_accepts"):
                            prev = self._last_counters.get(key, 0)
                            delta = counters.get(key, 0) - prev
                            # Clamp resets (server restart zeroes counters) to no-activity
                            if delta >= 0 and elapsed > 0:
                                result[f"{key}_rate"] = delta / elapsed
                            else:
                                result[f"{key}_rate"] = 0
                    else:
                        # First sample: rates are 0
                        for key in ("prompt_tokens_rate", "generated_tokens_rate", "speculative_accepts_rate"):
                            result[key] = 0
                    
                    # Compute rolling window averages
                    result.update(self._compute_averages(current_time, counters))
                    
                    self._last_counters = counters
                    self._last_timestamp = current_time
                    return result
        except Exception:
            pass
        
        # Unreachable server: drop baseline so reconnection re-baselines cleanly
        self._last_counters = None
        self._last_timestamp = None
        return self._empty()
    

    
    def _compute_averages(self, current_time: float, counters: dict) -> dict:
        """Compute rolling window averages for token rates.
        
        Returns dict with *_rate_avg keys computed as counter delta across
        the window divided by elapsed time.
        """
        # Find the oldest sample within the window
        window_start_time = current_time - self._window_seconds
        window_samples = [(t, c) for t, c in self._counter_window if t >= window_start_time]
        
        counter_keys = ("prompt_tokens", "generated_tokens", "speculative_accepts")
        return _compute_window_averages(window_samples, current_time, counter_keys)
    
    @staticmethod
    def _empty() -> dict:
        return {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "speculative_accepts": 0,
            "prompt_tokens_rate": 0,
            "generated_tokens_rate": 0,
            "speculative_accepts_rate": 0,
            "prompt_tokens_rate_avg": 0.0,
            "generated_tokens_rate_avg": 0.0,
            "speculative_accepts_rate_avg": 0.0,
        }
    
    def _parse_metrics(self, metrics_text: str) -> dict:
        """Parse Prometheus metrics text for llama-server counters.
        
        Values may be rendered in scientific notation (e.g. 3.05406e+06),
        so parse as float and convert to int.
        """
        patterns = {
            "prompt_tokens": r"llamacpp:prompt_tokens_total\s+([0-9.eE+\-]+)",
            "generated_tokens": r"llamacpp:tokens_predicted_total\s+([0-9.eE+\-]+)",
            "speculative_accepts": r"llamacpp:spec_decode_num_accepted_tokens_total\s+([0-9.eE+\-]+)",
        }
        
        counters = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, metrics_text)
            counters[key] = int(float(match.group(1))) if match else 0
        
        return counters


class RealOllamaStatsSource(OllamaStatsSource):
    """Collect ollama stats from debug logs.
    
    Ollama logs tokens in the format:
    llama_new_context: n_tokens = 64
    llama_sample: token=1234, prob=0.1234, logits=...
    
    We parse log files for prompt tokens (n_tokens) and generated tokens
    (counted from log entries with token data).
    """
    
    # Default paths where ollama debug logs may be located
    LOG_PATHS = [
        "/var/log/ollama.log",
        "/var/log/ollama/ollama.log",
        "/var/log/services/ollama.log",
    ]
    
    def __init__(self, log_path: str = None, window_seconds: float = 10.0):
        self._log_path = log_path
        self._last_prompt_tokens = 0
        self._last_generated_tokens = 0
        self._last_timestamp = None  # monotonic timestamp of last collect
        self._window_seconds = window_seconds
        # deque of (monotonic_ts, counters_dict) to track rolling window
        # ~1 sample/sec * window_seconds (window_seconds of 10 = 10 samples)
        self._counter_window = deque(maxlen=int(window_seconds))
    
    async def collect(self) -> dict:
        """Collect ollama stats from debug logs.
        
        Returns:
            dict with namespaced keys: ollama_prompt_tokens, ollama_generated_tokens,
            ollama_prompt_tokens_rate, ollama_generated_tokens_rate,
            ollama_prompt_tokens_rate_avg, ollama_generated_tokens_rate_avg
            Each value is the cumulative counter value or rate
        """
        prompt_tokens = 0
        generated_tokens = 0
        
        # Try to find a valid log path
        log_path = self._log_path
        if not log_path:
            for path in self.LOG_PATHS:
                try:
                    if await asyncio.to_thread(lambda p=path: __import__('os').path.exists(p)):
                        log_path = path
                        break
                except Exception:
                    continue
        
        if not log_path:
            return self._empty()
        
        try:
            # Read log file (limit to last 10000 lines to avoid unbounded growth)
            content = await asyncio.to_thread(self._read_log_file, log_path)
            
            # Parse prompt tokens from llama_new_context lines
            prompt_pattern = r'llama_new_context: n_tokens\s*=\s*(\d+)'
            prompt_matches = re.findall(prompt_pattern, content)
            if prompt_matches:
                # Use the last value (most recent context)
                prompt_tokens = int(prompt_matches[-1])
            
            # Parse generated tokens from log entries
            # Ollama logs each generated token in the format:
            # 2024/01/01 12:00:00 llama_token = 1234
            gen_pattern = r'llama_token\s*=\s*(\d+)'
            gen_matches = re.findall(gen_pattern, content)
            generated_tokens = len(gen_matches)
            
        except Exception:
            # If we can't read the log, return zeros
            pass
        
        current_time = asyncio.get_event_loop().time()
        
        # Track counter window for rolling average
        counters = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens
        }
        self._counter_window.append((current_time, counters))
        
        # Compute rates by differencing against previous sample, normalized to per-second
        if self._last_timestamp is not None:
            elapsed = current_time - self._last_timestamp
            if elapsed > 0:
                prompt_rate = max(0, (prompt_tokens - self._last_prompt_tokens) / elapsed)
                gen_rate = max(0, (generated_tokens - self._last_generated_tokens) / elapsed)
            else:
                prompt_rate = 0
                gen_rate = 0
        else:
            # First sample: no rate yet
            prompt_rate = 0
            gen_rate = 0
        
        # Update last values for next sample
        self._last_prompt_tokens = prompt_tokens
        self._last_generated_tokens = generated_tokens
        self._last_timestamp = current_time
        
        # Compute rolling window averages
        rate_avg = self._compute_ollama_averages(current_time, counters)
        
        return {
            "ollama_prompt_tokens": prompt_tokens,
            "ollama_generated_tokens": generated_tokens,
            "ollama_prompt_tokens_rate": prompt_rate,
            "ollama_generated_tokens_rate": gen_rate,
            "ollama_prompt_tokens_rate_avg": rate_avg["prompt"],
            "ollama_generated_tokens_rate_avg": rate_avg["generated"],
        }
    
    def _compute_ollama_averages(self, current_time: float, counters: dict) -> dict:
        """Compute rolling window averages for ollama token rates.
        
        Returns dict with rate_avg keys computed as counter delta across
        the window divided by elapsed time.
        """
        # Find the oldest sample within the window
        window_start_time = current_time - self._window_seconds
        window_samples = [(t, c) for t, c in self._counter_window if t >= window_start_time]
        
        counter_keys = ("prompt_tokens", "generated_tokens")
        averages = _compute_window_averages(window_samples, current_time, counter_keys)
        
        # Map from key_rate_avg to short key (e.g., prompt_tokens_rate_avg -> prompt)
        return {key.replace("_tokens", ""): averages.get(f"{key}_rate_avg", 0.0) for key in counter_keys}
    
    def _read_log_file(self, log_path: str) -> str:
        """Read and return log file contents.
        
        Reads the last 10000 lines to avoid unbounded growth over long-running servers.
        """
        with open(log_path, 'r') as f:
            # Read last 10000 lines to avoid reading entire file
            lines = f.readlines()
            content = ''.join(lines[-10000:])
        return content
    
    @staticmethod
    def _empty() -> dict:
        return {
            "ollama_prompt_tokens": 0,
            "ollama_generated_tokens": 0,
            "ollama_prompt_tokens_rate": 0,
            "ollama_generated_tokens_rate": 0,
            "ollama_prompt_tokens_rate_avg": 0.0,
            "ollama_generated_tokens_rate_avg": 0.0,
        }
