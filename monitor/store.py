"""SQLite database operations for telemetry storage."""

import aiosqlite
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path.home() / ".monitor" / "monitor.db"


async def init_db():
    """Initialize the database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
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
    process_memory=None
):
    """Insert a telemetry sample into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO telemetry_samples 
            (timestamp, memory_total, memory_free, memory_used, gpu_util, gpu_temp, gpu_power, process_memory)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                memory_total,
                memory_free,
                memory_used,
                gpu_util,
                gpu_temp,
                gpu_power,
                process_memory
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
