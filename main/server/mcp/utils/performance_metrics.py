# main/server/mcp/utils/performance_metrics.py
import psutil
from typing import Dict

async def get_metrics() -> Dict[str, float]:
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "active_users": 0  # Mock for now
    }
