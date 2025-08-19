from langchain.tools import Tool
from ..langchain.langchain_config import langchain_config
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool_name: str, func):
        try:
            self.tools[tool_name] = Tool(name=tool_name, func=func, description=f"Executes {tool_name}")
            langchain_config.register_adapter(tool_name, self.tools[tool_name])
            logger.info(f"Registered LangChain tool: {tool_name}")
        except Exception as e:
            error_logger.log_error("tool_register", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Tool registration failed: {str(e)}")
            raise

tool_manager = ToolManager()

# xAI Artifact Tags: #vial2 #mcp #langchain #tool #manager #neon_mcp
