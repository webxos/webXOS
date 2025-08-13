from langchain.prompts import PromptTemplate
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePrompt:
    def __init__(self):
        self.template = PromptTemplate(
            input_variables=["vial_id", "task"],
            template="Execute task for vial {vial_id}: {task}"
        )

    def format_prompt(self, vial_id: str, task: str) -> str:
        try:
            prompt = self.template.format(vial_id=vial_id, task=task)
            logger.info(f"Formatted prompt for vial {vial_id}: {prompt}")
            return prompt
        except Exception as e:
            logger.error(f"Prompt formatting error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Prompt formatting error: {str(e)}\n")
            raise
