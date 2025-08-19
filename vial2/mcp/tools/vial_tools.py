from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class VialTools:
    async def check_vial_status(self, vial_id: str):
        try:
            query = "SELECT event_data FROM vial_logs WHERE vial_id = $1 ORDER BY created_at DESC LIMIT 1"
            result = await neon_db.execute(query, vial_id)
            logger.info(f"Checked status for vial {vial_id}")
            return result
        except Exception as e:
            error_logger.log_error("vial_tools_check", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Vial tools check failed: {str(e)}")
            raise

# xAI Artifact Tags: #vial2 #mcp #tools #vial #vial_tools #neon_mcp
