from langchain.adapters.security import SecurityAdapter
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class ValidationChecker:
    def __init__(self):
        self.security_adapter = SecurityAdapter(api_key=os.getenv("AUTH_API_KEY", ""))

    def validate_input(self, input_data: str):
        try:
            if not self.security_adapter.validate_input(input_data):
                raise ValueError("Invalid input detected")
            logger.info("Input validation passed")
            return True
        except Exception as e:
            error_logger.log_error("input_validate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Input validation failed: {str(e)}")
            raise

validation_checker = ValidationChecker()

# xAI Artifact Tags: #vial2 #mcp #security #validation #checker #langchain #neon_mcp
