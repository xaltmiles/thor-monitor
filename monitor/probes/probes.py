"""High-level probe orchestration."""

from .interface import ServerDetector, ModelDetector, ProbeResult
from .server import ServerDetectorImpl
from .model import ModelDetectorImpl


async def run_probes(
    server_detector: ServerDetector = None,
    model_detector: ModelDetector = None
) -> ProbeResult:
    """
    Run server and model detection probes.
    
    Args:
        server_detector: Server detector instance (uses default if None)
        model_detector: Model detector instance (uses default if None)
        
    Returns:
        Probe result with servers, models, and warnings
    """
    if server_detector is None:
        server_detector = ServerDetectorImpl()
    if model_detector is None:
        model_detector = ModelDetectorImpl()
    
    # Detect servers
    servers = await server_detector.detect()
    
    # Detect models
    models = await model_detector.detect(servers)
    
    # Check for multi-model warning
    warning = None
    if len(models) > 1:
        model_names = [m.name for m in models]
        warning = f"Multiple loaded models detected: {', '.join(model_names)}"
    
    # Collect ollama guidance messages
    ollama_guidance = [m.guidance for m in models if m.server_type == "ollama" and m.guidance]
    
    return ProbeResult(
        servers=servers,
        models=models,
        warning=warning,
        ollama_guidance=ollama_guidance if ollama_guidance else None
    )
