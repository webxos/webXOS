import json
from fastapi import APIRouter
from ...mcp.error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/vial/status")
async def get_vial_status():
    try:
        status = {
            "vial1": {"status": "running", "latency": 50},
            "vial2": {"status": "stopped", "latency": 0},
            "vial3": {"status": "running", "latency": 75},
            "vial4": {"status": "stopped", "latency": 0}
        }
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": status}}
    except Exception as e:
        error_logger.log_error("vial_status", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Vial status fetch failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #netlify #vial_status #neon_mcp
