"""Real telemetry sources - collect from actual system."""

import asyncio
import re
from typing import Any
import psutil
import httpx
from .interface import MemorySource, GPUSource, ProcessSource, GPUMemorySource, LLaMAStatsSource


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
    """Collect llama-server metrics from Prometheus endpoint."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._last_counters = None
    
    async def collect(self) -> dict:
        """Collect llama-server metrics from /metrics endpoint.
        
        Returns:
            dict with keys: prompt_tokens, generated_tokens, speculative_accepts
            Each value is the cumulative counter value
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"http://{self.host}:{self.port}/metrics")
                
                if response.status_code == 200:
                    metrics_text = response.text
                    counters = self._parse_metrics(metrics_text)
                    
                    # Calculate rates (diff from last sample)
                    result = counters.copy()
                    
                    if self._last_counters:
                        for key in ["prompt_tokens", "generated_tokens", "speculative_accepts"]:
                            if key in counters and key in self._last_counters:
                                delta = counters[key] - self._last_counters[key]
                                # Rate per second (assuming 1 Hz sampling)
                                result[f"{key}_rate"] = delta
                    
                    self._last_counters = counters
                    return result
        except Exception:
            pass
        
        return {
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "speculative_accepts": 0,
            "prompt_tokens_rate": 0,
            "generated_tokens_rate": 0,
            "speculative_accepts_rate": 0
        }
    
    def _parse_metrics(self, metrics_text: str) -> dict:
        """Parse Prometheus metrics text for llama-server counters."""
        counters = {}
        
        # Patterns for llama-server Prometheus metrics
        patterns = {
            "prompt_tokens": r"llamacpp:prompt_tokens_total\s+(\d+)",
            "generated_tokens": r"llamacpp:tokens_generated_total\s+(\d+)",
            "speculative_accepts": r"llamacpp:speculative_accepts_total\s+(\d+)",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, metrics_text)
            if match:
                counters[key] = int(match.group(1))
            else:
                counters[key] = 0
        
        return counters
