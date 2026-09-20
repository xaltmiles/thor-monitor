"""Test fixtures for probe detection."""

import asyncio
from typing import Optional
from monitor.probes.interface import (
    ServerDetector, ModelDetector, 
    ServerInfo, ModelInfo, ProbeResult
)


class FixtureServerDetector(ServerDetector):
    """Fixture server detector with configurable results."""
    
    def __init__(self, servers: Optional[list[ServerInfo]] = None):
        self._servers = servers or []
    
    async def detect(self) -> list[ServerInfo]:
        """Return configured server list."""
        return self._servers.copy()


class FixtureModelDetector(ModelDetector):
    """Fixture model detector with configurable results."""
    
    def __init__(self, models: Optional[list[ModelInfo]] = None):
        self._models = models or []
    
    async def detect(self, servers: list[ServerInfo]) -> list[ModelInfo]:
        """Return configured model list."""
        return self._models.copy()


class FixtureProbeResult:
    """Fixture probe result for testing."""
    
    @staticmethod
    def create(servers=None, models=None, warning=None):
        """Create a probe result with fixture data."""
        from monitor.probes import ProbeResult
        
        return ProbeResult(
            servers=servers or [],
            models=models or [],
            warning=warning
        )
