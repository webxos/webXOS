from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import git
import json

logger = logging.getLogger(__name__)

class TrainingCheckpoint:
    def __init__(self):
        self.trainer = QuantumTrainer()
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def save_checkpoint(self, vial_id: str, commands: list):
        try:
            prompts = self.ds.generate_git_prompts(commands)
            checkpoint = await self.trainer.save_checkpoint(prompts)
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "training_checkpoint", json.dumps({"checkpoint": checkpoint, "vial": vial_id}))
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Saved final checkpoint for vial {vial_id}")
            logger.info(f"Saved checkpoint for vial {vial_id} with LangChain")
            return checkpoint
        except Exception as e:
            error_logger.log_error("training_checkpoint", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Training checkpoint failed: {str(e)}")
            raise

training_checkpoint = TrainingCheckpoint()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #checkpoint #git #neon_mcp
