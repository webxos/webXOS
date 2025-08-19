from langchain.chains.base import Chain
from ..tools.agent_manager import agent_manager
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class MCPChain(Chain):
    def __init__(self, agent_type: str):
        super().__init__()
        self.agent_type = agent_type
        self.agent = None

    async def _call(self, inputs: dict):
        try:
            self.agent = await agent_manager.get_agent(self.agent_type)
            if not self.agent:
                raise ValueError(f"No agent available for {self.agent_type}")
            response = await self.agent.run(inputs["query"])
            logger.info(f"MCP chain processed query for {self.agent_type}")
            return {"output": response}
        except Exception as e:
            error_logger.log_error("mcp_chain_call", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"MCP chain failed: {str(e)}")
            raise

    @property
    def input_keys(self):
        return ["query"]

    @property
    def output_keys(self):
        return ["output"]

mcp_chain = MCPChain("grok")

# xAI Artifact Tags: #vial2 #mcp #langchain #chain #neon #neon_mcp
