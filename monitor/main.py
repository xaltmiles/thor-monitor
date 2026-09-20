"""Main entry point for the monitor application."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monitor.store import init_db
from monitor.telemetry import RealMemorySource, RealGPUSource, RealProcessSource, RealGPUMemorySource, RealLLaMAStatsSource, RealOllamaStatsSource
from monitor.sampler import Sampler
from monitor.sessions import SessionTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run the monitor application."""
    logger.info("Initializing monitor...")
    
    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    
    # Create telemetry sources
    memory_source = RealMemorySource()
    gpu_source = RealGPUSource()
    process_source = RealProcessSource()
    gpu_memory_source = RealGPUMemorySource()
    llama_stats_source = RealLLaMAStatsSource()  # discovers the llama-server via probes
    ollama_stats_source = RealOllamaStatsSource()  # discovers ollama logs
    
    # Get sample rate from settings
    from .store import get_settings
    settings = await get_settings()
    # init_db always inserts the default row, so settings should never be None
    sample_rate = settings["sample_rate"]
    sampler_interval = 1.0 / sample_rate
    
    # Create and start sampler
    sampler = Sampler(
        sources=[memory_source, gpu_source, process_source, gpu_memory_source, llama_stats_source, ollama_stats_source],
        interval=sampler_interval,
        session_tracker=SessionTracker(),
    )
    
    # Start sampler in background
    sampler_task = asyncio.create_task(sampler.start())
    
    # Share the live sampler with the web app so settings updates can apply
    # without a restart (POST /api/settings -> sampler.set_interval).
    from .web.routes import app as web_app
    web_app.state.sampler = sampler
    
    # Start web server (routes.create_app owns the shared BenchmarkRunner)
    logger.info("Starting web server on 0.0.0.0:8000...")
    try:
        import uvicorn
        await uvicorn.Server(
            uvicorn.Config(
                "monitor.web.routes:app",
                host="0.0.0.0",
                port=8000,
                reload=False
            )
        ).serve()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await sampler.stop()
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass


def cli():
    """Synchronous entry point for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
