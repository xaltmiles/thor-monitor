"""Telemetry source interface - abstracts data collection from real sources."""

from abc import ABC, abstractmethod
from typing import Protocol


class TelemetrySource(Protocol):
    """Protocol for telemetry sources."""
    
    async def collect(self) -> dict:
        """Collect telemetry data from the source.
        
        Returns:
            dict with telemetry keys (e.g., 'memory_total', 'memory_free', etc.)
        """
        ...


class MemorySource(ABC):
    """Abstract base for memory telemetry sources."""
    
    @abstractmethod
    async def collect(self) -> dict:
        """Collect unified memory statistics.
        
        Returns:
            dict with keys: memory_total, memory_free, memory_used
        """
        ...


class GPUSource(ABC):
    """Abstract base for GPU telemetry sources."""
    
    @abstractmethod
    async def collect(self) -> dict:
        """Collect GPU statistics.
        
        Returns:
            dict with keys: gpu_util, gpu_temp, gpu_power
        """
        ...


class ProcessSource(ABC):
    """Abstract base for process table sources."""
    
    @abstractmethod
    async def collect(self) -> dict:
        """Collect process table data.
        
        Returns:
            dict with process information
        """
        ...


class GPUMemorySource(ABC):
    """Abstract base for per-process GPU memory attribution sources."""
    
    @abstractmethod
    async def collect(self) -> dict:
        """Collect per-process GPU memory attribution.
        
        Returns:
            dict with 'gpu_processes' key containing list of process GPU memory data
            Each process entry has: pid, name, gpu_memory
        """
        ...
