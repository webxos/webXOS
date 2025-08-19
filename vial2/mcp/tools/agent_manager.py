import os
from ..error_logging.error_log import error_logger
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.api_keys = {
            "grok": os.getenv("GROK_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", "")
        }

    async def register_agent(self, agent_type: str, credentials: Dict = None):
        try:
            if not credentials or not credentials.get("api_key"):
                if not self.api_keys.get(agent_type):
                    raise ValueError(f"No API key provided for {agent_type}")
                credentials = {"api_key": self.api_keys[agent_type]}
            self.agents[agent_type] = {"credentials": credentials["api_key"], "status": "active"}
            logger.info(f"Registered agent: {agent_type}")
        except Exception as e:
            error_logger.log_error("agent_register", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent registration failed: {str(e)}")
            raise

    async def get_agent(self, agent_type: str):
        agent = self.agents.get(agent_type)
        if agent and agent.get("status") == "active":
            return agent
        raise ValueError(f"Agent {agent_type} not available")

agent_manager = AgentManager()

# xAI Artifact Tags: #vial2 #mcp #tools #agent #manager #neon_mcp
