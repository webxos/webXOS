from fastapi import Request
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class MCPCompliance:
    def validate_request(self, request: Request):
        try:
            body = request.json()
            if not body or "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                raise ValueError("Invalid MCP request format")
            logger.info("MCP request validated")
            return True
        except Exception as e:
            error_logger.log_error("mcp_compliance_validate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"MCP validation failed: {str(e)}")
            raise

mcp_compliance = MCPCompliance()

# xAI Artifact Tags: #vial2 #mcp #compliance #neon #mcp_compliance #neon_mcp
