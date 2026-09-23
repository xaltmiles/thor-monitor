"""Catalog views - comparison table and timeline."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from ..store import (
    get_benchmark_runs, get_sessions_history, get_all_models,
    get_benchmark_run
)


async def get_comparison_data() -> Dict[str, Any]:
    """Get data for the comparison table.
    
    Returns:
        Dict with models list and run metrics
    """
    # Get all models to look up file_size
    all_models = await get_all_models()
    # Use a tuple of (name, quant, server_type) as key to avoid collision
    models_by_key = {(m.get('name'), m.get('quant'), m.get('server_type')): m for m in all_models}
    
    # Get all standard benchmark runs
    runs = await get_benchmark_runs(limit=1000)
    
    # Filter to standard runs only
    standard_runs = [r for r in runs if r.get("standard_run", False)]
    
    # Group by model (name + quant + server_type combination)
    # Use a tuple of (name, quant, server_type) as key to separate runs by server type
    model_runs: Dict[tuple, List[Dict]] = {}
    for run in standard_runs:
        model_key = (run.get('model_name', 'unknown'), run.get('model_quant', 'unknown'), run.get('server_type', 'unknown'))
        if model_key not in model_runs:
            model_runs[model_key] = []
        model_runs[model_key].append(run)
    
    # Build comparison data
    comparison_data = []
    for model_key, run_list in model_runs.items():
        model_name, model_quant, server_type = model_key  # Unpack tuple key
        context_length = run_list[0].get("context_length")
        
        # Look up file_size from models table
        file_size = None
        model_info = models_by_key.get(model_key)
        if model_info:
            file_size = model_info.get("file_size")
        
        # Calculate best/avg for each workload type
        workload_metrics = {}
        
        for run in run_list:
            workload_type = run.get("workload_type", "unknown")
            if workload_type not in workload_metrics:
                workload_metrics[workload_type] = []
            workload_metrics[workload_type].append(run)
        
        # For each workload, get best/avg values
        tok_s_metrics = {}
        for workload_type, runs in workload_metrics.items():
            # Use 'is not None' check to allow 0.0 as valid value
            tok_s_values = [r.get("gen_tok_s") for r in runs if r.get("gen_tok_s") is not None]
            if tok_s_values:
                tok_s_metrics[f"{workload_type}_tok_s"] = {
                    "best": max(tok_s_values),
                    "avg": sum(tok_s_values) / len(tok_s_values)
                }
            
            peak_gen_values = [r.get("peak_gen_tok_s") for r in runs if r.get("peak_gen_tok_s") is not None]
            if peak_gen_values:
                tok_s_metrics[f"{workload_type}_peak_gen_tok_s"] = {
                    "best": max(peak_gen_values),
                    "avg": sum(peak_gen_values) / len(peak_gen_values)
                }
        
        # Get memory footprint
        memory_footprint = None
        if file_size:
            memory_footprint = f"{file_size / (1024**3):.1f} GB"
        
        comparison_data.append({
            "model_name": model_name,
            "model_quant": model_quant,
            "context_length": context_length,
            "server_type": server_type,
            "memory_footprint": memory_footprint,
            "workload_metrics": tok_s_metrics,
        })
    
    # Sort by model name
    comparison_data.sort(key=lambda x: x["model_name"])
    
    # Collect all unique workload types for template rendering
    all_workload_types = set()
    for run_list in model_runs.values():
        for run in run_list:
            wt = run.get("workload_type")
            if wt:
                all_workload_types.add(wt)
    
    return {
        "models": comparison_data,
        "total_standard_runs": len(standard_runs),
        "workload_types": sorted(all_workload_types),
    }


async def get_timeline_data() -> Dict[str, Any]:
    """Get data for the timeline view.
    
    Returns:
        Dict with runs and sessions lists
    """
    # Get benchmark runs
    runs = await get_benchmark_runs(limit=1000)
    
    # Get sessions
    sessions = await get_sessions_history(limit=1000)
    
    # Get all models once (hoisted out of loop to fix N+1 issue)
    all_models = await get_all_models()
    models_by_id = {m.get("id"): m for m in all_models}
    
    # Build timeline events
    timeline_events = []
    
    for run in runs:
        event_type = "benchmark"
        workload_type = run.get("workload_type", "unknown")
        model_name = run.get("model_name", "unknown")
        server_type = run.get("server_type")
        timestamp = run.get("created_at", "")
        
        # Get tok/s info
        gen_tok_s = run.get("gen_tok_s")
        prompt_tok_s = run.get("prompt_tok_s")
        concurrent_throughput = run.get("concurrent_throughput")
        ttft = run.get("ttft")
        
        # Build details string with server type indicator
        server_indicator = f"{server_type} | " if server_type else ""
        
        details_parts = []
        if gen_tok_s:
            details_parts.append(f"gen: {gen_tok_s:.1f} tok/s")
        if prompt_tok_s:
            details_parts.append(f"prompt: {prompt_tok_s:.1f} tok/s")
        if concurrent_throughput:
            details_parts.append(f"concurrent: {concurrent_throughput:.1f} tok/s")
        if ttft:
            details_parts.append(f"TTFT: {ttft:.2f}s")
        
        details = ", ".join(details_parts) if details_parts else "benchmark run"
        details = f"{server_indicator}{details}"
        
        # Get memory footprint
        memory_footprint = None
        file_size = run.get("file_size")
        if file_size:
            memory_footprint = f"{file_size / (1024**3):.1f} GB"
        
        timeline_events.append({
            "type": event_type,
            "workload_type": workload_type,
            "model_name": model_name,
            "server_type": server_type,
            "timestamp": timestamp,
            "details": details,
            "memory_footprint": memory_footprint,
            "standard_run": run.get("standard_run", False),
        })
    
    for session in sessions:
        event_type = "session"
        model_id = session.get("model_id")
        start_time = session.get("start_time", "")
        end_time = session.get("end_time")
        avg_tok_s = session.get("avg_tok_s")
        total_tokens = session.get("total_tokens")
        
        # Get model name from pre-fetched models dict (N+1 fix)
        model_name = "unknown"
        if model_id in models_by_id:
            model_name = models_by_id[model_id].get("name", "unknown")
        
        details_parts = []
        if avg_tok_s:
            details_parts.append(f"avg: {avg_tok_s:.1f} tok/s")
        if total_tokens:
            details_parts.append(f"{total_tokens} tokens")
        
        details = ", ".join(details_parts) if details_parts else "session"
        
        timeline_events.append({
            "type": event_type,
            "model_name": model_name,
            "timestamp": start_time,
            "end_time": end_time,
            "details": details,
            "standard_run": True,  # Sessions are always "standard" observations
        })
    
    # Sort by timestamp (most recent first)
    timeline_events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {
        "events": timeline_events,
        "total_runs": len(runs),
        "total_sessions": len(sessions),
    }
