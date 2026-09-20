"""Main entry point for the monitor application."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monitor.store import init_db
from monitor.telemetry import RealMemorySource, RealGPUSource, RealProcessSource, RealGPUMemorySource
from monitor.sampler import Sampler

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
    
    # Create and start sampler
    sampler = Sampler(
        sources=[memory_source, gpu_source, process_source, gpu_memory_source],
        interval=1.0  # 1 Hz sampling
    )
    
    # Start sampler in background
    sampler_task = asyncio.create_task(sampler.start())
    
    # Start web server
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
