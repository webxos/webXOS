# main/server/mcp/utils/health_check.py
import os
import logging
from typing import Dict, Any
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")

class HealthCheck:
    async def get_health(self) -> Dict[str, Any]:
        try:
            return {
                "cpu_usage": 10.5,
                "memory_usage": 20.3,
                "active_users": 3,
                "balance": 500.0
            }
        except Exception as e:
            logger.error(f"Health check error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"Health check failed: {str(e)}")

    async def check_system(self) -> Dict[str, Any]:
        try:
            required_files = [
                "unified_server.py", "auth_manager.py", "webxos_balance.py", "error_handler.py",
                "logging.conf", "service_registry.py", "secrets_manager.py"
            ]
            file_status = {f: os.path.isfile(f"main/server/mcp/{f}") for f in required_files}
            return {"result": {"all_files_present": all(file_status.values()), "file_status": file_status}}
        except Exception as e:
            logger.error(f"System check error: {str(e)}", exc_info=True)
            raise MCPError(code=-32603, message=f"System check failed: {str(e)}")
