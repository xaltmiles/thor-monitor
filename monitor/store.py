"""SQLite database operations for telemetry storage."""

import aiosqlite
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import logging

DB_PATH = Path.home() / ".monitor" / "monitor.db"
logger = logging.getLogger(__name__)


# Default settings values - single source of truth
DEFAULTS = {
    "sample_rate": 1,
    "standard_workload_duration": 10,
    "max_queue_wait": 120,
    "warning_gpu_temp": 85.0,
}


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
                process_memory TEXT,
                gpu_process_memory TEXT,
                llama_stats TEXT,
                ollama_stats TEXT
            )
        """)
        
        # Create index on timestamp for efficient time-range queries
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_samples_timestamp 
            ON telemetry_samples(timestamp)
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
        
        # Check if ollama_stats column exists
        cursor = await db.execute("PRAGMA table_info(telemetry_samples)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "ollama_stats" not in column_names:
            logger.info("Adding ollama_stats column to telemetry_samples")
            await db.execute("""
                ALTER TABLE telemetry_samples 
                ADD COLUMN ollama_stats TEXT
            """)
            await db.commit()
        
        # Create index on timestamp for efficient time-range queries (migration for existing DBs)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_samples_timestamp 
            ON telemetry_samples(timestamp)
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
                peak_gen_tok_s REAL,
                concurrent_throughput REAL,
                model_name TEXT,
                model_quant TEXT,
                context_length INTEGER,
                server_type TEXT,
                server_port INTEGER,
                workload_params TEXT,
                tags TEXT,
                workload_results TEXT,
                memory_before TEXT,
                memory_during TEXT,
                memory_after TEXT,
                gpu_before TEXT,
                gpu_during TEXT,
                gpu_after TEXT,
                created_at TEXT NOT NULL,
                state TEXT,
                abort_reason TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Migrate stores created before later columns existed
        cursor = await db.execute("PRAGMA table_info(benchmark_runs)")
        br_columns = [col[1] for col in await cursor.fetchall()]
        for col in ("state", "abort_reason", "workload_results"):
            if col not in br_columns:
                logger.info("Adding %s column to benchmark_runs", col)
                await db.execute(f"ALTER TABLE benchmark_runs ADD COLUMN {col} TEXT")
        await db.commit()
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
                warning_gpu_temp REAL DEFAULT 85.0
            )
        """)
        
        # Insert default settings row if not exists
        await db.execute(
            "INSERT OR IGNORE INTO settings (id, sample_rate, standard_workload_duration, max_queue_wait, warning_gpu_temp) VALUES (1, ?, ?, ?, ?)",
            (DEFAULTS["sample_rate"], DEFAULTS["standard_workload_duration"], DEFAULTS["max_queue_wait"], DEFAULTS["warning_gpu_temp"])
        )
        await db.commit()
        
        # Migration: check for and add missing columns to benchmark_runs
        cursor = await db.execute("PRAGMA table_info(benchmark_runs)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "peak_gen_tok_s" not in column_names:
            logger.info("Adding peak_gen_tok_s column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN peak_gen_tok_s REAL
            """)
            await db.commit()
        
        if "model_name" not in column_names:
            logger.info("Adding model_name column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN model_name TEXT
            """)
            await db.commit()
        
        if "model_quant" not in column_names:
            logger.info("Adding model_quant column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN model_quant TEXT
            """)
            await db.commit()
        
        if "context_length" not in column_names:
            logger.info("Adding context_length column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN context_length INTEGER
            """)
            await db.commit()
        
        if "server_type" not in column_names:
            logger.info("Adding server_type column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN server_type TEXT
            """)
            await db.commit()
        
        if "workload_params" not in column_names:
            logger.info("Adding workload_params column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN workload_params TEXT
            """)
            await db.commit()
        
        if "tags" not in column_names:
            logger.info("Adding tags column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN tags TEXT
            """)
            await db.commit()
        
        if "server_port" not in column_names:
            logger.info("Adding server_port column to benchmark_runs")
            await db.execute("""
                ALTER TABLE benchmark_runs 
                ADD COLUMN server_port INTEGER
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
    llama_stats=None,
    ollama_stats=None
):
    """Insert a telemetry sample into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO telemetry_samples 
            (timestamp, memory_total, memory_free, memory_used, gpu_util, gpu_temp, gpu_power, process_memory, gpu_process_memory, llama_stats, ollama_stats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                llama_stats,
                ollama_stats
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
    """Get recent telemetry samples (legacy - returns all columns for backward compatibility)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM telemetry_samples ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_telemetry_history_for_plots(limit=100):
    """Get recent telemetry samples for plots page.
    
    Returns only columns needed for plots to minimize payload size:
    timestamp, gpu_util, gpu_temp, gpu_power, memory_used, memory_total,
    llama_stats, ollama_stats
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT timestamp, gpu_util, gpu_temp, gpu_power, 
                      memory_used, memory_total, llama_stats, ollama_stats
               FROM telemetry_samples 
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_telemetry_history_by_range(start_time: str, end_time: str = None, limit: int = None):
    """Get telemetry samples within a time range.
    
    Args:
        start_time: Start timestamp (ISO format)
        end_time: End timestamp (ISO format), optional
        limit: Maximum number of samples, optional
        
    Returns:
        List of telemetry samples with only plot-required columns
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        if end_time:
            if limit:
                async with db.execute(
                    """SELECT timestamp, gpu_util, gpu_temp, gpu_power, 
                              memory_used, memory_total, llama_stats, ollama_stats
                       FROM telemetry_samples 
                       WHERE timestamp >= ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (start_time, end_time, limit)
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with db.execute(
                    """SELECT timestamp, gpu_util, gpu_temp, gpu_power, 
                              memory_used, memory_total, llama_stats, ollama_stats
                       FROM telemetry_samples 
                       WHERE timestamp >= ? AND timestamp <= ?
                       ORDER BY timestamp DESC""",
                    (start_time, end_time)
                ) as cursor:
                    rows = await cursor.fetchall()
        else:
            if limit:
                async with db.execute(
                    """SELECT timestamp, gpu_util, gpu_temp, gpu_power, 
                              memory_used, memory_total, llama_stats, ollama_stats
                       FROM telemetry_samples 
                       WHERE timestamp >= ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (start_time, limit)
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with db.execute(
                    """SELECT timestamp, gpu_util, gpu_temp, gpu_power, 
                              memory_used, memory_total, llama_stats, ollama_stats
                       FROM telemetry_samples 
                       WHERE timestamp >= ?
                       ORDER BY timestamp DESC""",
                    (start_time,)
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


async def update_session(
    session_id: int,
    end_time: str = None,
    avg_tok_s: float = None,
    total_tokens: int = None
) -> None:
    """Update a session (typically to close it with its observed stats).
    
    Args:
        session_id: Session to update
        end_time: Session end timestamp (ISO format)
        avg_tok_s: Observed average tokens per second
        total_tokens: Total tokens processed during the session
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE sessions
                SET end_time = COALESCE(?, end_time),
                    avg_tok_s = COALESCE(?, avg_tok_s),
                    total_tokens = COALESCE(?, total_tokens)
                WHERE id = ?""",
            (end_time, avg_tok_s, total_tokens, session_id)
        )
        await db.commit()


async def insert_benchmark_run(
    model_id: int,
    workload_type: str,
    standard_run: bool,
    total_time: float = None,
    ttft: float = None,
    prompt_tok_s: float = None,
    gen_tok_s: float = None,
    peak_gen_tok_s: float = None,
    concurrent_throughput: float = None,
    model_name: str = None,
    model_quant: str = None,
    context_length: int = None,
    server_type: str = None,
    server_port: int = None,
    state: str = None,
    workload_params: str = None,
    tags: str = None,
    workload_results: str = None,
    memory_before: str = None,
    memory_during: str = None,
    memory_after: str = None,
    gpu_before: str = None,
    gpu_during: str = None,
    gpu_after: str = None
) -> int:
    """Insert a benchmark run into the database.
    
    Args:
        model_id: ID of the model being benchmarked
        workload_type: Type of workload (e.g., 'standard', 'custom')
        standard_run: Whether this is a standard run
        total_time: Total time to complete the workload in seconds
        ttft: Time to first token in seconds
        prompt_tok_s: Prompt processing rate in tokens/second
        gen_tok_s: Generation rate in tokens/second
        peak_gen_tok_s: Peak generation rate in tokens/second
        concurrent_throughput: Concurrent throughput in tokens/second
        model_name: Name of the model
        model_quant: Quantization type
        context_length: Context length in tokens
        server_type: Type of server (ollama, llama-server)
        workload_params: JSON string of workload parameters
        tags: Comma-separated tags (e.g., 'standard,short-workload')
        memory_before: JSON string of memory samples before run
        memory_during: JSON string of memory samples during run
        memory_after: JSON string of memory samples after run
        gpu_before: JSON string of GPU samples before run
        gpu_during: JSON string of GPU samples during run
        gpu_after: JSON string of GPU samples after run
        
    Returns:
        Benchmark run ID
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO benchmark_runs 
                (model_id, workload_type, standard_run, total_time, ttft,
                 prompt_tok_s, gen_tok_s, peak_gen_tok_s, concurrent_throughput,
                 model_name, model_quant, context_length, server_type,
                server_port, state, workload_params, tags, workload_results,
                 memory_before, memory_during, memory_after,
                 gpu_before, gpu_during, gpu_after, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, workload_type, standard_run, total_time, ttft,
             prompt_tok_s, gen_tok_s, peak_gen_tok_s, concurrent_throughput,
             model_name, model_quant, context_length, server_type,
             server_port, state, workload_params, tags, workload_results,
             memory_before, memory_during, memory_after,
             gpu_before, gpu_during, gpu_after,
             datetime.now(timezone.utc).isoformat())
        )
        await db.commit()
        
        # Return the inserted ID
        async with db.execute(
            "SELECT id FROM benchmark_runs ORDER BY id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def update_benchmark_run(
    run_id: int,
    state: str = None,
    abort_reason: str = None,
    total_time: float = None,
    ttft: float = None,
    prompt_tok_s: float = None,
    gen_tok_s: float = None,
    peak_gen_tok_s: float = None,
    concurrent_throughput: float = None,
    workload_results: str = None,
    memory_before: str = None,
    memory_during: str = None,
    gpu_before: str = None,
    gpu_during: str = None,
    memory_after: str = None,
    gpu_after: str = None
) -> None:
    """Update a benchmark run (typically to update with final metrics and after-run samples).
    
    Args:
        run_id: Benchmark run to update
        total_time: Total time to complete the workload in seconds
        ttft: Time to first token in seconds
        prompt_tok_s: Prompt processing rate in tokens/second
        gen_tok_s: Generation rate in tokens/second
        peak_gen_tok_s: Peak generation rate in tokens/second
        concurrent_throughput: Concurrent throughput in tokens/second
        memory_during: JSON string of memory samples during run
        gpu_during: JSON string of GPU samples during run
        memory_after: JSON string of memory samples after run
        gpu_after: JSON string of GPU samples after run
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE benchmark_runs
                SET state = COALESCE(?, state),
                    abort_reason = COALESCE(?, abort_reason),
                    total_time = COALESCE(?, total_time),
                    ttft = COALESCE(?, ttft),
                    prompt_tok_s = COALESCE(?, prompt_tok_s),
                    gen_tok_s = COALESCE(?, gen_tok_s),
                    peak_gen_tok_s = COALESCE(?, peak_gen_tok_s),
                    concurrent_throughput = COALESCE(?, concurrent_throughput),
                    workload_results = COALESCE(?, workload_results),
                    memory_before = COALESCE(?, memory_before),
                    memory_during = COALESCE(?, memory_during),
                    gpu_before = COALESCE(?, gpu_before),
                    gpu_during = COALESCE(?, gpu_during),
                    memory_after = COALESCE(?, memory_after),
                    gpu_after = COALESCE(?, gpu_after)
                WHERE id = ?""",
            (state, abort_reason, total_time, ttft, prompt_tok_s, gen_tok_s,
             peak_gen_tok_s, concurrent_throughput, workload_results, memory_before, memory_during,
             gpu_before, gpu_during, memory_after, gpu_after, run_id)
        )
        await db.commit()


async def get_benchmark_run(run_id: int) -> Optional[dict]:
    """Get a specific benchmark run.
    
    Args:
        run_id: ID of the benchmark run
        
    Returns:
        Benchmark run info dict or None if not found
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM benchmark_runs WHERE id = ?",
            (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_benchmark_runs(limit: int = 100) -> list[dict]:
    """Get recent benchmark runs.
    
    Args:
        limit: Maximum number of runs to return
        
    Returns:
        List of benchmark run info dicts
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_settings() -> Optional[dict]:
    """Get current settings.
    
    Returns:
        Settings dict with all config values, or None if no settings exist.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM settings WHERE id = 1") as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_settings(
    sample_rate: int = None,
    standard_workload_duration: int = None,
    max_queue_wait: int = None,
    warning_gpu_temp: float = None
) -> Optional[dict]:
    """Update settings. Only provided parameters are updated.
    
    Args:
        sample_rate: Sampling rate in Hz
        standard_workload_duration: Standard short workload duration in seconds
        max_queue_wait: Maximum time to wait for server to drain in seconds
        warning_gpu_temp: GPU temperature warning threshold in Celsius
    
    Returns:
        Updated settings dict, or None if no settings exist
    """
    # Build UPDATE clause dynamically
    updates = []
    values = []
    
    if sample_rate is not None:
        updates.append("sample_rate = ?")
        values.append(sample_rate)
    if standard_workload_duration is not None:
        updates.append("standard_workload_duration = ?")
        values.append(standard_workload_duration)
    if max_queue_wait is not None:
        updates.append("max_queue_wait = ?")
        values.append(max_queue_wait)
    if warning_gpu_temp is not None:
        updates.append("warning_gpu_temp = ?")
        values.append(warning_gpu_temp)
    if not updates:
        # No updates, just fetch current
        return await get_settings()
    
    values.append(1)  # WHERE id = 1
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE settings SET {', '.join(updates)} WHERE id = ?",
            values
        )
        await db.commit()
    
    return await get_settings()
