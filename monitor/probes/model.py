"""Model detection via server APIs."""

import asyncio
import re
from pathlib import Path
from typing import Optional
import httpx

from .interface import ModelDetector, ModelInfo, ProbeError, ServerInfo


class ModelDetectorImpl(ModelDetector):
    """Detect loaded models by querying server APIs."""
    
    def __init__(self, client_timeout: float = 5.0):
        self._client_timeout = client_timeout
    
    async def detect(self, servers: list[ServerInfo]) -> list[ModelInfo]:
        """
        Detect loaded model for each server.
        
        Args:
            servers: List of detected servers
            
        Returns:
            List of loaded model information
        """
        models = []
        
        for server in servers:
            model = None
            
            if server.type == "ollama":
                model = await self._detect_ollama(server)
            elif server.type == "llama-server":
                model = await self._detect_llama_server(server)
            
            if model:
                model.server_type = server.type
                models.append(model)
        
        return models
    
    async def _detect_ollama(self, server: ServerInfo) -> Optional[ModelInfo]:
        """Detect ollama loaded model via API."""
        port = server.port or 11434
        
        # Try running-models endpoint first
        try:
            async with httpx.AsyncClient(timeout=self._client_timeout) as client:
                response = await client.get(f"http://localhost:{port}/api/running-models")
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    
                    if models:
                        model_name = models[0]
                        # Parse model name to extract quant and context
                        info = self._parse_ollama_model_name(model_name)
                        info.server_type = "ollama"
                        return info
        except (httpx.RequestError, ValueError):
            pass
        
        # Fallback: try /api/ps for process list (may be empty)
        # For now, return None since we can't reliably detect loaded model
        # This is acceptable for v1 - the model info will be detected from llama-server
        return None
    
    async def _detect_llama_server(self, server: ServerInfo) -> Optional[ModelInfo]:
        """Detect llama-server loaded model via cmdline and optional API."""
        # Extract model path from cmdline
        model_path = None
        for arg in server.cmdline or []:
            if arg.startswith("--model="):
                model_path = arg.split("=", 1)[1]
                break
            elif arg == "--model" and server.cmdline:
                idx = server.cmdline.index(arg)
                if idx + 1 < len(server.cmdline):
                    model_path = server.cmdline[idx + 1]
                    break
        
        if not model_path:
            return None
        
        # Parse model filename to extract info
        model_name = Path(model_path).stem
        info = self._parse_llama_model_name(model_name, model_path)
        info.server_type = "llama-server"
        
        # Try to get additional info from /info endpoint (if available)
        port = server.port or 8080
        try:
            async with httpx.AsyncClient(timeout=self._client_timeout) as client:
                response = await client.get(f"http://localhost:{port}/info")
                if response.status_code == 200:
                    data = response.json()
                    if "model_context_length" in data:
                        info.context_length = data["model_context_length"]
                    if "model_parameters" in data:
                        # Estimate quant from parameter count
                        params = data["model_parameters"]
                        info.quant = self._estimate_quant(params)
        except (httpx.RequestError, ValueError):
            pass
        
        return info
    
    def _parse_ollama_model_name(self, name: str) -> ModelInfo:
        """Parse ollama model name to extract info."""
        # ollama model names: "model:quant" or "model:tag"
        # Examples: "llama3:8b", "mistral:7b-instruct-v0.2-q4_K_M"
        
        parts = name.split(":")
        model_name = parts[0]
        quant = None
        
        if len(parts) > 1:
            quant = parts[1]
        
        # Extract context length if present (e.g., "llama3-70b:70b")
        context_length = None
        context_match = re.search(r'(\d+)b', quant or model_name, re.IGNORECASE)
        if context_match:
            context_val = int(context_match.group(1))
            if context_val > 100:  # Likely context length, not parameter count
                context_length = context_val * 1024  # Convert to tokens
        
        return ModelInfo(
            name=model_name,
            quant=quant,
            context_length=context_length
        )
    
    def _parse_llama_model_name(self, name: str, path: str) -> ModelInfo:
        """Parse llama-server model name to extract info."""
        # llama-server model names often contain quant info
        # Examples: "mistral-7b-instruct-v0.2.Q4_K_M.gguf", "llama-2-7b.Q2_K.gguf"
        
        quant = None
        context_length = None
        
        # Extract quant from filename
        quant_patterns = [
            r'Q\d+[_K_M]*',  # Q4_K_M, Q2_K, etc.
            r'fp\d+',         # fp16, fp32
        ]
        
        for pattern in quant_patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                quant = match.group(0)
                break
        
        # Extract context length from filename
        context_match = re.search(r'(\d+)k', name, re.IGNORECASE)
        if context_match:
            context_length = int(context_match.group(1)) * 1024
        else:
            # Try parameter count
            param_match = re.search(r'(\d+)b', name, re.IGNORECASE)
            if param_match:
                params = int(param_match.group(1))
                # Estimate context length (common values)
                context_estimates = {
                    7: 4096,
                    13: 8192,
                    70: 8192,
                }
                context_length = context_estimates.get(params)
        
        # Get file size
        file_size = None
        try:
            file_size = Path(path).stat().st_size
        except (OSError, IOError):
            pass
        
        return ModelInfo(
            name=name,
            quant=quant,
            context_length=context_length,
            file_size=file_size
        )
    
    def _estimate_quant(self, params: int) -> Optional[str]:
        """Estimate quant from parameter count."""
        # Rough estimation - actual quant depends on bits per weight
        # Common quantizations: 4-bit, 5-bit, 6-bit, 8-bit
        if params:
            # This is a rough estimate; actual quant needs to be parsed from model file
            return None
        return None
