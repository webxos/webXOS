import os
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents = {}
        self.api_keys = {
            "grok": os.getenv("GROK_API_KEY", "")
        }

    async def register_agent(self, agent_type: str, credentials: dict):
        try:
            if not credentials.get("api_key") and not self.api_keys.get(agent_type):
                raise ValueError("API key required for agent registration")
            self.agents[agent_type] = {"credentials": credentials.get("api_key") or self.api_keys[agent_type]}
            logger.info(f"Registered agent: {agent_type}")
        except Exception as e:
            error_logger.log_error("agent_register", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent registration failed: {str(e)}")
            raise

    async def get_agent(self, agent_type: str):
        return self.agents.get(agent_type)

agent_manager = AgentManager()

# xAI Artifact Tags: #vial2 #mcp #tools #agent #manager #neon_mcp
