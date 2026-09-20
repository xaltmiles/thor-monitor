"""Test fixtures for telemetry sources."""

from monitor.telemetry.interface import TelemetrySource, MemorySource, GPUSource, ProcessSource


class FixtureMemorySource(MemorySource):
    """Fixture memory source with configurable values."""
    
    def __init__(self, total: int, free: int, used: int):
        self.total = total
        self.free = free
        self.used = used
    
    async def collect(self) -> dict:
        """Return fixture memory data."""
        return {
            "memory_total": self.total,
            "memory_free": self.free,
            "memory_used": self.used
        }


class FixtureGPUSource(GPUSource):
    """Fixture GPU source with configurable values."""
    
    def __init__(self, util: float, temp: float, power: float):
        self.util = util
        self.temp = temp
        self.power = power
    
    async def collect(self) -> dict:
        """Return fixture GPU data."""
        return {
            "gpu_util": self.util,
            "gpu_temp": self.temp,
            "gpu_power": self.power
        }


class FixtureProcessSource(ProcessSource):
    """Fixture process source with configurable values."""
    
    def __init__(self, processes: list):
        self.processes = processes
    
    async def collect(self) -> dict:
        """Return fixture process data."""
        return {"processes": self.processes}


class FixtureTelemetrySource:
    """Fixture telemetry source combining multiple fixture sources."""
    
    def __init__(
        self,
        memory_source: MemorySource,
        gpu_source: GPUSource,
        process_source: ProcessSource
    ):
        self.memory_source = memory_source
        self.gpu_source = gpu_source
        self.process_source = process_source
    
    async def collect(self) -> dict:
        """Collect from all fixture sources."""
        memory = await self.memory_source.collect()
        gpu = await self.gpu_source.collect()
        process = await self.process_source.collect()
        
        return {
            **memory,
            **gpu,
            **process
        }
