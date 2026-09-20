"""Server detection via process scanning."""

import asyncio
import re
from typing import Optional
import psutil

from .interface import ServerDetector, ServerInfo, ProbeError


class ServerDetectorImpl(ServerDetector):
    """Detect running servers by scanning process table."""
    
    def __init__(self):
        self._process_cache = None
        self._cache_time = None
    
    async def detect(self) -> list[ServerInfo]:
        """
        Detect which servers are running.
        
        Returns:
            List of detected server information
        """
        servers = []
        
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                info = proc.info
                pid = info.get("pid")
                name = info.get("name")
                cmdline = info.get("cmdline")
                
                if cmdline is None:
                    continue
                
                cmdline_str = " ".join(cmdline)
                
                # Check for ollama serve
                if "ollama" in name.lower() or "ollama" in cmdline_str:
                    if "serve" in cmdline_str:
                        port = self._extract_port(cmdline)
                        servers.append(ServerInfo(
                            pid=pid,
                            name="ollama",
                            type="ollama",
                            port=port,
                            cmdline=cmdline
                        ))
                        continue
                
                # Check for llama-server
                if "llama-server" in name.lower() or "llama-server" in cmdline_str:
                    port = self._extract_port(cmdline)
                    servers.append(ServerInfo(
                        pid=pid,
                        name="llama-server",
                        type="llama-server",
                        port=port,
                        cmdline=cmdline
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        return servers
    
    def _extract_port(self, cmdline: list[str]) -> Optional[int]:
        """Extract port from command line arguments."""
        for i, arg in enumerate(cmdline):
            if arg in ("--port", "-p", "--server-port"):
                if i + 1 < len(cmdline):
                    try:
                        return int(cmdline[i + 1])
                    except ValueError:
                        continue
            # Handle --port=8080 format
            if arg.startswith("--port="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    continue
            if arg.startswith("-p") and len(arg) > 2:
                try:
                    return int(arg[2:])
                except ValueError:
                    continue
        return None
