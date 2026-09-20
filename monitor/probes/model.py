"""Model detection via server APIs."""

import asyncio
import re
from pathlib import Path
from typing import Optional
import httpx

from .interface import ModelDetector, ModelInfo, ProbeError, ServerInfo


class ModelDetectorImpl(ModelDetector):
    """Detect loaded models by querying server APIs."""
    
    def __init__(self, client_timeout: float = 5.0, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._client_timeout = client_timeout
        self._transport = transport
    
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
            if server.type == "ollama":
                models.extend(await self._detect_ollama(server))
            elif server.type == "llama-server":
                model = await self._detect_llama_server(server)
                if model:
                    model.server_type = server.type
                    models.append(model)
        
        return models
    
    async def _detect_ollama(self, server: ServerInfo) -> list[ModelInfo]:
        """Detect ollama loaded models via its running-model API (GET /api/ps).
        
        Ollama can have multiple models resident at once, so return all of them.
        
        Debug logging detection: ollama's /api/ps response includes a "details"
        object with a "streaming" field when debug logging is enabled.
        If "streaming" is False or missing, ollama is not logging tokens,
        which means we can't track passive tok/s.
        """
        port = server.port or 11434
        models = []
        
        try:
            async with httpx.AsyncClient(timeout=self._client_timeout, transport=self._transport) as client:
                response = await client.get(f"http://localhost:{port}/api/ps")
                
                if response.status_code == 200:
                    data = response.json()
                    for entry in data.get("models", []):
                        if not isinstance(entry, dict):
                            continue
                        name = entry.get("name") or entry.get("model")
                        if not name:
                            continue
                        details = entry.get("details") or {}
                        
                        # Check if debug logging is enabled (streaming = True means logs enabled)
                        streaming = details.get("streaming", False)
                        guidance = None
                        if not streaming:
                            guidance = (
                                "Ollama debug logging not enabled. "
                                "Run 'systemctl edit ollama' and add: "
                                "[Service]\nEnvironment=OLLAMA_DEBUG=1\n"
                            )
                        
                        models.append(ModelInfo(
                            name=name,
                            quant=details.get("quantization_level") or None,
                            context_length=entry.get("context_length"),
                            file_size=entry.get("size"),
                            server_type="ollama",
                            guidance=guidance,
                        ))
        except (httpx.RequestError, ValueError):
            pass
        
        return models
    
    async def _detect_llama_server(self, server: ServerInfo) -> Optional[ModelInfo]:
        """Detect llama-server loaded model via cmdline and optional API."""
        cmdline = server.cmdline or []
        # llama.cpp accepts -m/--model for the model path
        model_path = self._arg_value(cmdline, ["-m", "--model"])
        
        if not model_path:
            return None
        
        # Display name: --alias if given, else the model filename stem
        alias = self._arg_value(cmdline, ["--alias"])
        model_name = alias or Path(model_path).stem
        
        # Ground-truth context length from -c/--ctx-size when present
        ctx_value = self._arg_value(cmdline, ["-c", "--ctx-size"])
        context_override = None
        if ctx_value is not None:
            try:
                context_override = int(ctx_value)
            except ValueError:
                pass
        
        info = self._parse_llama_model_name(model_name, model_path, context_override)
        # The display name may be an alias with no quant info; fall back to the file stem
        if info.quant is None:
            info.quant = self._quant_from_name(Path(model_path).stem)
        info.server_type = "llama-server"
        
        # Try to get additional info from /info endpoint (if available)
        port = server.port or 8080
        try:
            async with httpx.AsyncClient(timeout=self._client_timeout, transport=self._transport) as client:
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
    
    @staticmethod
    def _arg_value(cmdline: list[str], flags: list[str]) -> Optional[str]:
        """Get the value for the first matching flag in a command line.
        
        Handles '--flag value', '--flag=value' and attached short forms like '-c4096'.
        """
        for i, arg in enumerate(cmdline):
            if arg in flags:
                if i + 1 < len(cmdline):
                    return cmdline[i + 1]
            for flag in flags:
                if arg.startswith(flag + "="):
                    return arg.split("=", 1)[1]
                # Attached short form (e.g. -c4096): flag is a single dash + one char
                if len(flag) == 2 and flag.startswith("-") and arg.startswith(flag) and len(arg) > 2:
                    return arg[2:]
        return None
    
    @staticmethod
    def _quant_from_name(name: str) -> Optional[str]:
        """Extract a quantization tag from a model name/filename."""
        for pattern in (r"Q\d+[_K_M]*", r"fp\d+"):
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _parse_llama_model_name(self, name: str, path: str, context_override: Optional[int] = None) -> ModelInfo:
        """Parse llama-server model name to extract info."""
        # llama-server model names often contain quant info
        # Examples: "mistral-7b-instruct-v0.2.Q4_K_M.gguf", "llama-2-7b.Q2_K.gguf"
        
        quant = None
        context_length = None
        
        # Extract quant from name
        quant = self._quant_from_name(name)
        
        # Extract context length from filename
        if context_override is not None:
            context_length = context_override
        else:
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
