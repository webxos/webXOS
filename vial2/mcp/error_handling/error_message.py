from mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ErrorMessage:
    def format_error(self, error: Exception, context: str):
        try:
            message = f"Error in {context}: {str(error)}. See logs for details."
            logger.error(message)
            error_logger.log_error("error_format", str(error), str(error.__traceback__), sql_statement=None, sql_error_code=None, params={context})
            return message
        except Exception as e:
            error_logger.log_error("error_format_fail", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Error formatting failed: {str(e)}")
            return "An unexpected error occurred. Please try again."

error_message = ErrorMessage()

# xAI Artifact Tags: #vial2 #mcp #error #handling #message #neon_mcp
