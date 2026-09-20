"""Test fixtures for telemetry sources."""

from monitor.telemetry.interface import TelemetrySource, MemorySource, GPUSource, ProcessSource, GPUMemorySource, LLaMAStatsSource


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


class FixtureGPUMemorySource(GPUMemorySource):
    """Fixture GPU memory source with configurable per-process attribution."""
    
    def __init__(self, processes: list = None):
        self.processes = processes or []
    
    async def collect(self) -> dict:
        """Return fixture GPU process data."""
        return {"gpu_processes": self.processes}


class FixtureLLaMAStatsSource(LLaMAStatsSource):
    """Fixture LLaMA stats source with configurable values."""
    
    def __init__(
        self,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        speculative_accepts: int = 0,
        prompt_tokens_rate: float = 0.0,
        generated_tokens_rate: float = 0.0,
        speculative_accepts_rate: float = 0.0
    ):
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.speculative_accepts = speculative_accepts
        self.prompt_tokens_rate = prompt_tokens_rate
        self.generated_tokens_rate = generated_tokens_rate
        self.speculative_accepts_rate = speculative_accepts_rate
    
    async def collect(self) -> dict:
        """Return fixture LLaMA stats data."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "speculative_accepts": self.speculative_accepts,
            "prompt_tokens_rate": self.prompt_tokens_rate,
            "generated_tokens_rate": self.generated_tokens_rate,
            "speculative_accepts_rate": self.speculative_accepts_rate
        }


class FixtureOllamaStatsSource:
    """Fixture Ollama stats source with configurable values.
    
    Note: Keys are namespaced (ollama_prompt_tokens, etc.) to avoid
    collision with llama_stats when FixtureTelemetrySource.collect() merges them.
    """
    
    def __init__(
        self,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        prompt_tokens_rate: float = 0.0,
        generated_tokens_rate: float = 0.0
    ):
        self._prompt_tokens = prompt_tokens
        self._generated_tokens = generated_tokens
        self._prompt_tokens_rate = prompt_tokens_rate
        self._generated_tokens_rate = generated_tokens_rate
    
    async def collect(self) -> dict:
        """Return fixture Ollama stats data with namespaced keys."""
        return {
            "ollama_prompt_tokens": self._prompt_tokens,
            "ollama_generated_tokens": self._generated_tokens,
            "ollama_prompt_tokens_rate": self._prompt_tokens_rate,
            "ollama_generated_tokens_rate": self._generated_tokens_rate
        }


class FixtureTelemetrySource:
    """Fixture telemetry source combining multiple fixture sources."""
    
    def __init__(
        self,
        memory_source: MemorySource,
        gpu_source: GPUSource,
        gpu_memory_source: "FixtureGPUMemorySource" = None,
        llama_stats_source: "FixtureLLaMAStatsSource" = None,
        ollama_stats_source: "FixtureOllamaStatsSource" = None,
        process_source: ProcessSource = None
    ):
        self.memory_source = memory_source
        self.gpu_source = gpu_source
        self.gpu_memory_source = gpu_memory_source or FixtureGPUMemorySource([])
        self.llama_stats_source = llama_stats_source or FixtureLLaMAStatsSource()
        self.ollama_stats_source = ollama_stats_source or FixtureOllamaStatsSource()
        self.process_source = process_source or FixtureProcessSource([])
    
    async def collect(self) -> dict:
        """Collect from all fixture sources."""
        memory = await self.memory_source.collect()
        gpu = await self.gpu_source.collect()
        gpu_memory = await self.gpu_memory_source.collect()
        llama_stats = await self.llama_stats_source.collect()
        ollama_stats = await self.ollama_stats_source.collect()
        process = await self.process_source.collect()
        
        # Namespace ollama stats to avoid key collision with llama_stats
        # The database only has llama_stats column, so we store both
        # in the same field with namespaced keys
        ollama_stats_namespaced = {}
        for key, value in ollama_stats.items():
            if key in ("prompt_tokens", "generated_tokens", "prompt_tokens_rate", "generated_tokens_rate"):
                ollama_stats_namespaced[f"ollama_{key}"] = value
            else:
                ollama_stats_namespaced[key] = value
        
        return {
            **memory,
            **gpu,
            **gpu_memory,
            **llama_stats,
            **ollama_stats_namespaced,
            **process
        }
