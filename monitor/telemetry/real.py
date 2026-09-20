"""Real telemetry sources - collect from actual system."""

import asyncio
from typing import Any
import psutil
from .interface import MemorySource, GPUSource, ProcessSource


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
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(
                None,
                lambda: psutil.Popen(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", 
                     "--format=csv,noheader,nounits"],
                    stdout=-1, stderr=-1
                )
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
