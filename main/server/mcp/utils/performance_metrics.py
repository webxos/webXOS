# main/server/mcp/utils/performance_metrics.py
import psutil
import logging
from typing import Dict, Any
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")

class PerformanceMetrics:
    async def get_metrics(self) -> Dict[str, Any]:
        try:
            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "active_users": 5,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Metrics error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Metrics retrieval failed: {str(e)}")
