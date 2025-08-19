from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import json
import os

logger = logging.getLogger(__name__)

class OfflineHandler:
    def __init__(self):
        self.offline_queue = []

    def queue_offline_request(self, request: dict):
        try:
            self.offline_queue.append(request)
            with open("offline_queue.json", "w") as f:
                json.dump(self.offline_queue, f)
            logger.info("Queued offline request")
        except Exception as e:
            error_logger.log_error("offline_queue", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Offline queuing failed: {str(e)}")
            raise

offline_handler = OfflineHandler()

# xAI Artifact Tags: #vial2 #mcp #offline #handler #neon_mcp
