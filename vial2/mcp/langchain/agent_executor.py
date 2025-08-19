from langchain.agents import AgentExecutor
from ..langchain.tool_manager import tool_manager
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class AgentExecutor:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.executor = None

    async def execute(self, query: str):
        try:
            tools = [tool for name, tool in tool_manager.tools.items() if name.startswith(self.agent_type)]
            self.executor = AgentExecutor.from_agent_and_tools(tools=tools)
            result = await self.executor.run(query)
            logger.info(f"Executed agent {self.agent_type} with query: {query}")
            return result
        except Exception as e:
            error_logger.log_error("agent_execute", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent execution failed: {str(e)}")
            raise

agent_executor = AgentExecutor("grok")

# xAI Artifact Tags: #vial2 #mcp #langchain #agent #executor #neon_mcp
