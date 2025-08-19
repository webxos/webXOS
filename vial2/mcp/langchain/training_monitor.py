from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import git
import time

logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self):
        self.trainer = QuantumTrainer()
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def monitor_training(self, vial_id: str, commands: list):
        try:
            prompts = self.ds.generate_git_prompts(commands)
            status = await self.trainer.monitor(prompts)
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "training_monitor", {"status": status, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "vial": vial_id})
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Monitored training for vial {vial_id} with final status {status}")
            logger.info(f"Monitored training for vial {vial_id} with LangChain")
            return status
        except Exception as e:
            error_logger.log_error("training_monitor", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Training monitoring failed: {str(e)}")
            raise

training_monitor = TrainingMonitor()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #monitor #git #neon_mcp
