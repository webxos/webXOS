from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..langchain.training_orchestrator import training_orchestrator
from ..error_logging.error_log import error_logger
import logging
import asyncio
import git

logger = logging.getLogger(__name__)

class TrainingScheduler:
    def __init__(self):
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def schedule_training(self, vial_id: str, commands: list, interval: int):
        try:
            while True:
                await training_orchestrator.orchestrate_training(vial_id, commands)
                self.repo.index.add(["*"])
                self.repo.index.commit(f"Scheduled training for vial {vial_id}")
                logger.info(f"Scheduled training for vial {vial_id}")
                await asyncio.sleep(interval)
        except Exception as e:
            error_logger.log_error("training_schedule", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Training scheduling failed: {str(e)}")
            raise

training_scheduler = TrainingScheduler()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #scheduler #git #neon_mcp
