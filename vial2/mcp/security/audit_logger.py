from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import time
import json

logger = logging.getLogger(__name__)

async def log_audit_event(event_type: str, details: dict):
    try:
        query = "INSERT INTO audit_logs (event_type, details, timestamp) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
        await neon_db.execute(query, event_type, json.dumps(details), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        logger.info(f"Audit event logged: {event_type}")
    except Exception as e:
        error_logger.log_error("audit_log", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={})
        logger.error(f"Audit logging failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #security #audit #logger #neon_mcp
