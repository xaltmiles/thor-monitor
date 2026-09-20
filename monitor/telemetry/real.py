"""Real telemetry sources - collect from actual system."""

import asyncio
import re
from typing import Any
import psutil
import httpx
from .interface import MemorySource, GPUSource, ProcessSource, GPUMemorySource, LLaMAStatsSource
from ..probes.server import ServerDetectorImpl


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
    """
    
    DETECT_CACHE_SECONDS = 10.0
    
    def __init__(self, host: str = None, port: int = None, timeout: float = 2.0):
        self.host = host  # None -> discovered from probes
        self.port = port  # None -> discovered from probes
        self.timeout = timeout
        self._last_counters = None
        self._detect_cache = None  # (monotonic_time, host, port)
    
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
            speculative_accepts) and per-second rates (*_rate) computed by
            differencing against the previous sample (1 Hz sampling).
        """
        host, port = await self._resolve_target()
        if not host or not port:
            self._last_counters = None
            return self._empty()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"http://{host}:{port}/metrics")
                
                if response.status_code == 200:
                    counters = self._parse_metrics(response.text)
                    
                    result = counters.copy()
                    if self._last_counters:
                        for key in ("prompt_tokens", "generated_tokens", "speculative_accepts"):
                            prev = self._last_counters.get(key, 0)
                            delta = counters.get(key, 0) - prev
                            # Clamp resets (server restart zeroes counters) to no-activity
                            result[f"{key}_rate"] = max(0, delta)
                    
                    self._last_counters = counters
                    return result
        except Exception:
            pass
        
        # Unreachable server: drop baseline so reconnection re-baselines cleanly
        self._last_counters = None
        return self._empty()
    
    @staticmethod
    def _empty() -> dict:
        return {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "speculative_accepts": 0,
            "prompt_tokens_rate": 0,
            "generated_tokens_rate": 0,
            "speculative_accepts_rate": 0,
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
