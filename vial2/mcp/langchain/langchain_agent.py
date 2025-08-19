from langchain.agents import initialize_agent
from ..langchain.tool_manager import tool_manager
from ..langchain.memory_manager import memory_manager
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class LangChainAgent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent = None

    async def initialize(self):
        try:
            tools = [tool for name, tool in tool_manager.tools.items() if name.startswith(self.agent_type)]
            self.agent = initialize_agent(tools, memory=memory_manager.get_memory("vial1"))
            logger.info(f"Initialized LangChain agent: {self.agent_type}")
        except Exception as e:
            error_logger.log_error("agent_init", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent initialization failed: {str(e)}")
            raise

langchain_agent = LangChainAgent("grok")

# xAI Artifact Tags: #vial2 #mcp #langchain #agent #neon #neon_mcp
