"""Session tracking - recognize periods of real usage and record them.

A Session is a period during which a server (llama-server or ollama) is actively
generating tokens. The tracker watches the server metrics stream
(from the telemetry sources) and opens/closes sessions in the store with
observed speed stats.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Union

from .store import insert_session, update_session, insert_model

logger = logging.getLogger(__name__)


class SessionTracker:
    """Detect usage sessions from llama-server or ollama token counters.
    
    A session opens when the generation rate rises above zero and closes
    after the server has been idle for IDLE_TIMEOUT seconds.
    
    Supports both llama-server (via Prometheus metrics with *_rate keys)
    and ollama (via log-based stats with *_rate keys).
    """
    
    IDLE_TIMEOUT = 30.0  # seconds of inactivity before a session closes
    
    def __init__(self, model_name: str = None, idle_timeout: float = None):
        self.model_name = model_name
        self.idle_timeout = idle_timeout if idle_timeout is not None else self.IDLE_TIMEOUT
        self._session_id: Optional[int] = None
        self._start_monotonic: Optional[float] = None
        self._last_activity_monotonic: Optional[float] = None
        self._tokens_generated: int = 0
        self._tokens_prompt: int = 0
    
    async def observe(self, stats: Optional[dict], server_type: str = "llama-server") -> None:
        """Feed one telemetry sample into the tracker.
        
        Args:
            stats: dict from the stats source with cumulative counters and *_rate keys;
                None when no stats available. For llama-server, keys are
                generated_tokens_rate, prompt_tokens_rate. For ollama, same keys
                (or ollama_generated_tokens_rate, ollama_prompt_tokens_rate).
            server_type: "llama-server" or "ollama"
        """
        now = time.monotonic()
        
        if not stats:
            await self._maybe_close(now)
            return
        
        # Check for ollama stats first (namespaced keys), then llama stats
        gen_rate = stats.get("ollama_generated_tokens_rate") or stats.get("generated_tokens_rate") or 0
        prompt_rate = stats.get("ollama_prompt_tokens_rate") or stats.get("prompt_tokens_rate") or 0
        active = (gen_rate + prompt_rate) > 0
        
        if active:
            if self._session_id is None:
                await self._open(stats, now, server_type)
            self._last_activity_monotonic = now
            self._tokens_generated += int(gen_rate)
            self._tokens_prompt += int(prompt_rate)
        
        await self._maybe_close(now)
    
    async def _open(self, stats: dict, now: float, server_type: str) -> None:
        """Open a new session, resolving the model row for the FK."""
        model_id = await insert_model(
            name=self.model_name,
            server_type=server_type,
        )
        self._session_id = await insert_session(
            model_id=model_id,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self._start_monotonic = now
        self._last_activity_monotonic = now
        self._tokens_generated = 0
        self._tokens_prompt = 0
        logger.info("Session opened (id=%s, model=%s, server_type=%s)", self._session_id, self.model_name, server_type)
    
    async def _maybe_close(self, now: float) -> None:
        """Close the active session once the server has been idle long enough."""
        if self._session_id is None:
            return
        idle_for = now - (self._last_activity_monotonic or now)
        if idle_for < self.idle_timeout:
            return
        
        duration = max(0.0, (self._last_activity_monotonic or now) - (self._start_monotonic or now))
        avg_tok_s = (self._tokens_generated / duration) if duration > 0 else None
        total = self._tokens_generated + self._tokens_prompt
        
        await update_session(
            self._session_id,
            end_time=datetime.now(timezone.utc).isoformat(),
            avg_tok_s=round(avg_tok_s, 2) if avg_tok_s is not None else None,
            total_tokens=total,
        )
        logger.info(
            "Session closed (id=%s, %.1fs, %d tok, avg %s tok/s)",
            self._session_id, duration, total, avg_tok_s,
        )
        self._session_id = None
        self._start_monotonic = None
        self._last_activity_monotonic = None
        self._tokens_generated = 0
        self._tokens_prompt = 0
