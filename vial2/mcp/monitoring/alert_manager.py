from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time

router = APIRouter()

logger = logging.getLogger(__name__)

class AlertRequest(BaseModel):
    vial_id: str

@router.get("/mcp/api/vial/alerts")
async def get_alerts(request: AlertRequest):
    try:
        query = "SELECT COUNT(*) FROM vial_logs WHERE event_type = 'error' AND created_at > NOW() - INTERVAL '1 hour' AND vial_id = $1"
        count = await neon_db.execute(query, request.vial_id)
        alerts = {"error_count": count, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if int(count) > 5:
            logger.warning(f"High error count detected: {count}")
        encoded_response = jsonable_encoder({"jsonrpc": "2.0", "result": {"status": "success", "data": alerts}})
        logger.info("Alert data retrieved with JSON encoding")
        return encoded_response
    except Exception as e:
        error_logger.log_error("alert_manager", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={request.vial_id})
        logger.error(f"Alert retrieval failed: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}}

# xAI Artifact Tags: #vial2 #mcp #monitoring #alert #manager #json #neon_mcp
