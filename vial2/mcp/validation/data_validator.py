from ..error_logging.error_log import error_logger
import logging
import re

logger = logging.getLogger(__name__)

class DataValidator:
    def validate_vial_id(self, vial_id: str):
        try:
            if not vial_id or not re.match(r'^[a-zA-Z0-9_-]{1,50}$', vial_id):
                raise ValueError("Invalid vial_id format")
            logger.info(f"Validated vial_id: {vial_id}")
            return True
        except Exception as e:
            error_logger.log_error("data_validate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Data validation failed: {str(e)}")
            raise

data_validator = DataValidator()

# xAI Artifact Tags: #vial2 #mcp #validation #data #validator #neon_mcp
