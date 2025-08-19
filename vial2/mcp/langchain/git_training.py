from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class GitTraining:
    def __init__(self):
        self.trainer = QuantumTrainer()
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def train_with_git(self, vial_id: str, git_commands: list):
        try:
            prompts = self.ds.generate_git_prompts(git_commands)
            await self.trainer.train(prompts)
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "git_training", {"commands": git_commands})
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Trained vial {vial_id} with Git commands")
            logger.info(f"Trained vial {vial_id} with Git and LangChain")
        except Exception as e:
            error_logger.log_error("git_train", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Git training failed: {str(e)}")
            raise

git_training = GitTraining()

# xAI Artifact Tags: #vial2 #mcp #langchain #git #training #neon_mcp
