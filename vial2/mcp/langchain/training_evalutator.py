from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class TrainingEvaluator:
    def __init__(self):
        self.trainer = QuantumTrainer()
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def evaluate_training(self, vial_id: str, test_data: list):
        try:
            prompts = self.ds.generate_git_prompts(test_data)
            score = await self.trainer.evaluate(prompts)
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "training_evaluation", {"score": score})
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Evaluated training for vial {vial_id}")
            logger.info(f"Evaluated training for vial {vial_id} with LangChain")
            return score
        except Exception as e:
            error_logger.log_error("training_evaluate", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Training evaluation failed: {str(e)}")
            raise

training_evaluator = TrainingEvaluator()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #evaluator #git #neon_mcp
