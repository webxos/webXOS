from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import time
import json

logger = logging.getLogger(__name__)

class TrainingLogger:
    async def log_training(self, vial_id: str, training_data: list):
        try:
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            log_data = {"data": training_data, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            await neon_db.execute(query, vial_id, "training_log", json.dumps(log_data))
            logger.info(f"Logged training for vial {vial_id}")
        except Exception as e:
            error_logger.log_error("training_log", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Training logging failed: {str(e)}")
            raise

training_logger = TrainingLogger()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #logger #neon_mcp
