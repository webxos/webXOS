from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..langchain.training_deployer import training_deployer
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class TrainingFinalizer:
    def __init__(self):
        self.ds = DataSynthesizer()
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def finalize_training(self, vial_id: str, commands: list):
        try:
            prompts = self.ds.generate_git_prompts(commands)
            model = await training_deployer.deploy_training(vial_id, prompts)
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Finalized training for vial {vial_id} with model {model.id}")
            logger.info(f"Finalized training for vial {vial_id} with LangChain")
            return {"status": "completed", "model_id": model.id}
        except Exception as e:
            error_logger.log_error("training_finalize", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Training finalization failed: {str(e)}")
            raise

training_finalizer = TrainingFinalizer()

# xAI Artifact Tags: #vial2 #mcp #langchain #training #finalizer #git #neon_mcp
