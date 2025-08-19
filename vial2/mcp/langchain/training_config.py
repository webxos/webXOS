from langchain.trainers import QuantumTrainerConfig
from ..ds import DataSynthesizer
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class TrainingConfig:
    def __init__(self):
        self.config = QuantumTrainerConfig(epochs=10, learning_rate=0.01)
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def configure_training(self, agent_type: str, data: list):
        try:
            prompts = self.ds.generate_git_prompts(data)
            config = self.config.update({"training_data": prompts})
            self.repo.index.add(["training_config.py"])
            self.repo.index.commit(f"Updated training config for {agent_type}")
            logger.info(f"Configured training for {agent_type} with Git integration")
            return config
        except Exception as e:
            error_logger.log_error("training_config", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Training configuration failed: {str(e)}")
            raise

training_config = TrainingConfig()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #config #git #neon_mcp
