from mcp.error_handling.error_message import error_message
from mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ErrorDisplay:
    def show_error(self, error: Exception, context: str):
        try:
            message = error_message.format_error(error, context)
            logger.error(f"Displayed error: {message}")
            return message
        except Exception as e:
            error_logger.log_error("error_display", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Error display failed: {str(e)}")
            return "Failed to display error."

error_display = ErrorDisplay()

# xAI Artifact Tags: #vial2 #mcp #ui #error #display #neon_mcp
