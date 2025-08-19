from fastapi import APIRouter, Depends
from ..database.neon_connection import neon_db
from ..security.octokit_oauth import get_octokit_auth
from ..error_logging.error_log import error_logger
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.get("/mcp/api/vial/data")
async def get_vial_data(vial_id: str, token: str = Depends(get_octokit_auth)):
    try:
        query = "SELECT event_data FROM vial_logs WHERE vial_id = $1 ORDER BY created_at DESC LIMIT 1"
        result = await neon_db.execute(query, vial_id)
        logger.info(f"Fetched data for vial {vial_id}")
        return {"jsonrpc": "2.0", "result": {"status": "success", "data": result}}
    except Exception as e:
        error_logger.log_error("vial_data_fetch", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
        logger.error(f"Vial data fetch failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #api #endpoints #neon #neon_mcp
