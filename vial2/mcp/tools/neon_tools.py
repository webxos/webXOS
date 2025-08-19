from .vial_tools import VialTools
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class NeonTools(VialTools):
    async def sync_vial_data(self, vial_id: str):
        try:
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "sync", {"status": "success"})
            logger.info(f"Synced vial data for {vial_id}")
        except Exception as e:
            error_logger.log_error("neon_tools_sync", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Neon tools sync failed: {str(e)}")
            raise

# xAI Artifact Tags: #vial2 #mcp #tools #neon #neon_tools #neon_mcp
