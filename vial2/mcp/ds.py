from langchain.prompts import PromptTemplate
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class DataSynthesizer:
    def __init__(self):
        self.template = PromptTemplate(input_variables=["command"], template="Train model with Git command: {command}")

    def generate_git_prompts(self, commands: list):
        try:
            prompts = [self.template.format(command=cmd) for cmd in commands]
            logger.info("Generated Git prompts for training")
            return prompts
        except Exception as e:
            error_logger.log_error("ds_prompt", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Prompt generation failed: {str(e)}")
            raise

ds = DataSynthesizer()

# xAI Artifact Tags: #vial2 #mcp #ds #prompting #langchain #git #neon_mcp
