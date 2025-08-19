from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..langchain.training_evaluator import training_evaluator
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class TrainingOptimizer:
    def __init__(self):
        self.trainer = QuantumTrainer()
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def optimize_training(self, vial_id: str, commands: list):
        try:
            prompts = self.ds.generate_git_prompts(commands)
            score = await training_evaluator.evaluate_training(vial_id, prompts)
            if score < 0.8:
                self.trainer.adjust_hyperparameters(learning_rate=0.005, epochs=15)
                await self.trainer.train(prompts)
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Optimized training for vial {vial_id} with score {score}")
            logger.info(f"Optimized training for vial {vial_id} with LangChain")
            return score
        except Exception as e:
            error_logger.log_error("training_optimize", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Training optimization failed: {str(e)}")
            raise

training_optimizer = TrainingOptimizer()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #optimizer #git #neon_mcp
