"""Sampler - collects telemetry at 1 Hz and stores it."""

import asyncio
import logging
import json
from typing import Sequence
from datetime import datetime

from .store import insert_telemetry_sample
from .telemetry.interface import TelemetrySource, LLaMAStatsSource

logger = logging.getLogger(__name__)


class Sampler:
    """Sampler that collects telemetry data at a fixed interval."""
    
    def __init__(self, sources: Sequence[TelemetrySource], interval: float = 1.0):
        """
        Initialize the sampler.
        
        Args:
            sources: Sequence of telemetry sources to collect from
            interval: Collection interval in seconds (default: 1.0)
        """
        self.sources = sources
        self.interval = interval
        self._running = False
        self._task = None
    
    async def start(self):
        """Start the sampler loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Sampler started")
    
    async def stop(self):
        """Stop the sampler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sampler stopped")
    
    async def _run_loop(self):
        """Main sampling loop."""
        while self._running:
            start_time = asyncio.get_event_loop().time()
            
            try:
                sample = await self._collect_sample()
                await insert_telemetry_sample(**sample)
            except Exception as e:
                logger.error(f"Error collecting sample: {e}")
            
            # Sleep for the remaining time
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _collect_sample(self) -> dict:
        """Collect from all sources and merge results."""
        sample = {}
        
        for source in self.sources:
            try:
                data = await source.collect()
                # Map processes to process_memory for db storage
                if "processes" in data:
                    sample["process_memory"] = json.dumps(data["processes"])
                # Map GPU processes to gpu_process_memory for db storage
                elif "gpu_processes" in data:
                    sample["gpu_process_memory"] = json.dumps(data["gpu_processes"])
                # Map LLaMA stats to llama_stats for db storage
                elif "prompt_tokens" in data or "generated_tokens_rate" in data:
                    sample["llama_stats"] = json.dumps(data)
                else:
                    sample.update(data)
            except Exception as e:
                logger.warning(f"Error collecting from {source}: {e}")
        
        return sample
