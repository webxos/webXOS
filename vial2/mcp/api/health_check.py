from fastapi import APIRouter
from ...mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/vial/health")
async def health_check():
    try:
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": {"uptime": "active", "timestamp": "2025-08-18T20:48:00Z"}}}
    except Exception as e:
        error_logger.log_error("health_check", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Health check failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #api #health #check #neon_mcp
