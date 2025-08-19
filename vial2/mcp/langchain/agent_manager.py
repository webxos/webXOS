from langchain.trainers import QuantumTrainer
from mcp.langchain.training_orchestrator import training_orchestrator
from mcp.error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

class AgentManager:
    async def train_agent(self, agent_type: str, training_data: list):
        try:
            for vial_id in ["vial1", "vial2"]:  # Multi-vial support
                await training_orchestrator.orchestrate_training(vial_id, training_data)
            logger.info(f"Trained agent {agent_type} across multiple vials")
        except Exception as e:
            error_logger.log_error("agent_train", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent training failed: {str(e)}")
            raise

agent_manager = AgentManager()

# xAI Artifact Tags: #vial2 #mcp #langchain #agent #manager #neon_mcp
