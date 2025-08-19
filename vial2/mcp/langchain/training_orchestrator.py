from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..langchain.agent_manager import agent_manager
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    def __init__(self):
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def orchestrate_training(self, vial_id: str, commands: list):
        try:
            prompts = self.ds.generate_git_prompts(commands)
            await agent_manager.train_agent("grok", prompts)
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Orchestrated training for vial {vial_id}")
            logger.info(f"Orchestrated training for vial {vial_id} with LangChain")
        except Exception as e:
            error_logger.log_error("training_orchestrate", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Training orchestration failed: {str(e)}")
            raise

training_orchestrator = TrainingOrchestrator()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #orchestrator #git #neon_mcp
