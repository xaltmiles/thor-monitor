"""SQLite database operations for telemetry storage."""

import aiosqlite
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import logging

DB_PATH = Path.home() / ".monitor" / "monitor.db"
logger = logging.getLogger(__name__)


async def init_db():
    """Initialize the database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
        # Create telemetry_samples table if not exists
        await db.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                memory_total INTEGER,
                memory_free INTEGER,
                memory_used INTEGER,
                gpu_util REAL,
                gpu_temp REAL,
                gpu_power REAL,
                process_memory TEXT
            )
        """)
        
        # Check if gpu_process_memory column exists
        cursor = await db.execute("PRAGMA table_info(telemetry_samples)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "gpu_process_memory" not in column_names:
            logger.info("Adding gpu_process_memory column to telemetry_samples")
            await db.execute("""
                ALTER TABLE telemetry_samples 
                ADD COLUMN gpu_process_memory TEXT
            """)
            await db.commit()
        
        # Check if llama_stats column exists
        cursor = await db.execute("PRAGMA table_info(telemetry_samples)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "llama_stats" not in column_names:
            logger.info("Adding llama_stats column to telemetry_samples")
            await db.execute("""
                ALTER TABLE telemetry_samples 
                ADD COLUMN llama_stats TEXT
            """)
            await db.commit()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                quant TEXT,
                context_length INTEGER,
                server_type TEXT,
                file_size INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                workload_type TEXT NOT NULL,
                standard_run BOOLEAN NOT NULL,
                total_time REAL,
                ttft REAL,
                prompt_tok_s REAL,
                gen_tok_s REAL,
                concurrent_throughput REAL,
                memory_before TEXT,
                memory_during TEXT,
                memory_after TEXT,
                gpu_before TEXT,
                gpu_during TEXT,
                gpu_after TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                avg_tok_s REAL,
                total_tokens INTEGER,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                sample_rate INTEGER DEFAULT 1,
                standard_workload_duration INTEGER DEFAULT 10,
                max_queue_wait INTEGER DEFAULT 120,
                warning_gpu_temp REAL DEFAULT 85.0,
                warning_gpu_util REAL DEFAULT 95.0
            )
        """)
        await db.commit()


async def insert_telemetry_sample(
    memory_total=None,
    memory_free=None,
    memory_used=None,
    gpu_util=None,
    gpu_temp=None,
    gpu_power=None,
    process_memory=None,
    gpu_process_memory=None,
    llama_stats=None
):
    """Insert a telemetry sample into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO telemetry_samples 
            (timestamp, memory_total, memory_free, memory_used, gpu_util, gpu_temp, gpu_power, process_memory, gpu_process_memory, llama_stats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                memory_total,
                memory_free,
                memory_used,
                gpu_util,
                gpu_temp,
                gpu_power,
                process_memory,
                gpu_process_memory,
                llama_stats
            )
        )
        await db.commit()


async def get_latest_telemetry():
    """Get the most recent telemetry sample."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM telemetry_samples ORDER BY timestamp DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_telemetry_history(limit=100):
    """Get recent telemetry samples."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM telemetry_samples ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def insert_model(
    name: str,
    quant: str = None,
    context_length: int = None,
    server_type: str = None,
    file_size: int = None
) -> int:
    """Insert or update a model in the database.
    
    Args:
        name: Model name
        quant: Quantization type
        context_length: Context length in tokens
        server_type: Type of server (ollama, llama-server)
        file_size: Model file size in bytes
        
    Returns:
        Model ID
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # Check if model already exists
        async with db.execute(
            "SELECT id FROM models WHERE name = ? AND server_type = ?",
            (name, server_type)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Update existing model
                await db.execute(
                    """UPDATE models SET 
                        quant = ?, context_length = ?, file_size = ?, 
                        created_at = ? 
                        WHERE id = ?""",
                    (quant, context_length, file_size, 
                     datetime.now(timezone.utc).isoformat(), row[0])
                )
                await db.commit()
                return row[0]
        
        # Insert new model
        await db.execute(
            """INSERT INTO models 
                (name, quant, context_length, server_type, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (name, quant, context_length, server_type, file_size,
             datetime.now(timezone.utc).isoformat())
        )
        await db.commit()
        
        # Return the inserted ID
        async with db.execute(
            "SELECT id FROM models WHERE name = ? AND server_type = ?",
            (name, server_type)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_loaded_model(server_type: str = None) -> Optional[dict]:
    """Get the currently loaded model.
    
    Args:
        server_type: Optional server type filter
        
    Returns:
        Model info dict or None if no model loaded
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        if server_type:
            async with db.execute(
                "SELECT * FROM models WHERE server_type = ? ORDER BY created_at DESC LIMIT 1",
                (server_type,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
        else:
            async with db.execute(
                "SELECT * FROM models ORDER BY created_at DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None


async def get_all_models() -> list[dict]:
    """Get all models in the database.
    
    Returns:
        List of model info dicts
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM models ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def insert_session(
    model_id: int,
    start_time: str,
    end_time: str = None,
    avg_tok_s: float = None,
    total_tokens: int = None
) -> int:
    """Insert a session into the database.
    
    Args:
        model_id: ID of the model being served
        start_time: Session start timestamp (ISO format)
        end_time: Session end timestamp (ISO format), optional
        avg_tok_s: Average tokens per second, optional
        total_tokens: Total tokens processed, optional
        
    Returns:
        Session ID
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO sessions 
                (model_id, start_time, end_time, avg_tok_s, total_tokens)
                VALUES (?, ?, ?, ?, ?)""",
            (model_id, start_time, end_time, avg_tok_s, total_tokens)
        )
        await db.commit()
        
        # Return the inserted ID
        async with db.execute(
            "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_active_session() -> Optional[dict]:
    """Get the currently active session (without end_time).
    
    Returns:
        Session info dict or None if no active session
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sessions WHERE end_time IS NULL ORDER BY start_time DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_sessions_history(limit: int = 100) -> list[dict]:
    """Get recent sessions.
    
    Args:
        limit: Maximum number of sessions to return
        
    Returns:
        List of session info dicts
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
