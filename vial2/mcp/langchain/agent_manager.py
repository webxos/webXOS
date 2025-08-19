from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..error_logging.error_log import error_logger
import logging
import os
import git

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.trainers = {}
        self.api_keys = {"grok": os.getenv("GROK_API_KEY", "")}
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def train_agent(self, agent_type: str, training_data: list):
        try:
            if not self.api_keys.get(agent_type):
                raise ValueError(f"No API key for {agent_type}")
            ds = DataSynthesizer()
            prompts = ds.generate_git_prompts(training_data)
            self.trainers[agent_type] = QuantumTrainer(api_key=self.api_keys[agent_type])
            await self.trainers[agent_type].train(prompts)
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Trained agent {agent_type} with Git prompts")
            logger.info(f"Trained agent {agent_type} with LangChain and Git")
        except Exception as e:
            error_logger.log_error("agent_train", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Agent training failed: {str(e)}")
            raise

agent_manager = AgentManager()

# xAI Artifact Tags: #vial2 #mcp #langchain #agent #manager #training #git #neon_mcp
