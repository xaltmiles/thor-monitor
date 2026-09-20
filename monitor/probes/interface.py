"""Probe interfaces - abstract server and model detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerInfo:
    """Information about a detected server."""
    pid: int
    name: str
    type: str  # "ollama" or "llama-server"
    port: Optional[int] = None
    cmdline: Optional[list] = None


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    quant: Optional[str] = None
    context_length: Optional[int] = None
    file_size: Optional[int] = None
    server_type: Optional[str] = None
    guidance: Optional[str] = None  # User guidance (e.g., "enable debug logging")


@dataclass
class ProbeResult:
    """Result of a probe operation."""
    servers: list[ServerInfo]
    models: list[ModelInfo]
    warning: Optional[str] = None
    ollama_guidance: Optional[list[str]] = None  # Ollama-specific guidance messages


class ProbeError(Exception):
    """Error during probing."""
    pass


class ServerDetector(ABC):
    """Detector for running servers via process scan."""
    
    @abstractmethod
    async def detect(self) -> list[ServerInfo]:
        """
        Detect which servers are running.
        
        Returns:
            List of detected server information
        """
        pass


class ModelDetector(ABC):
    """Detector for loaded models via API/probing."""
    
    @abstractmethod
    async def detect(self, servers: list[ServerInfo]) -> list[ModelInfo]:
        """
        Detect loaded model for each server.
        
        Args:
            servers: List of detected servers
            
        Returns:
            List of loaded model information
        """
        pass
