"""Telemetry sources - collect system and server statistics."""

from .interface import TelemetrySource, MemorySource, GPUSource, ProcessSource, GPUMemorySource, LLaMAStatsSource, OllamaStatsSource
from .real import RealMemorySource, RealGPUSource, RealProcessSource, RealGPUMemorySource, RealLLaMAStatsSource, RealOllamaStatsSource
from .fixtures import (
    FixtureMemorySource,
    FixtureGPUSource,
    FixtureProcessSource,
    FixtureGPUMemorySource,
    FixtureTelemetrySource,
    FixtureLLaMAStatsSource,
    FixtureOllamaStatsSource,
)

__all__ = [
    "TelemetrySource",
    "MemorySource",
    "GPUSource",
    "ProcessSource",
    "GPUMemorySource",
    "LLaMAStatsSource",
    "OllamaStatsSource",
    "RealMemorySource",
    "RealGPUSource",
    "RealProcessSource",
    "RealGPUMemorySource",
    "RealLLaMAStatsSource",
    "RealOllamaStatsSource",
    "FixtureMemorySource",
    "FixtureGPUSource",
    "FixtureProcessSource",
    "FixtureGPUMemorySource",
    "FixtureTelemetrySource",
    "FixtureLLaMAStatsSource",
    "FixtureOllamaStatsSource",
]
