from langchain.trainers import QuantumTrainer
from ..ds import DataSynthesizer
from ..error_logging.error_log import error_logger
import logging
import git

logger = logging.getLogger(__name__)

class MCPChain:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.trainer = None
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    async def train_chain(self, training_data: list):
        try:
            ds = DataSynthesizer()
            prompts = ds.generate_git_prompts(training_data)
            self.trainer = QuantumTrainer()
            await self.trainer.train(prompts)
            self.repo.index.add(["*"])
            self.repo.index.commit(f"Trained MCP chain for {self.agent_type} with Git prompts")
            logger.info(f"Trained MCP chain for agent {self.agent_type} with LangChain")
        except Exception as e:
            error_logger.log_error("chain_train", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"MCP chain training failed: {str(e)}")
            raise

mcp_chain = MCPChain("grok")

# xAI Artifact Tags: #vial2 #mcp #langchain #chain #training #git #neon_mcp
