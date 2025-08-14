import logging
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptRequest(BaseModel):
    context: str
    task_type: str

class BasePrompt:
    """Generates base prompts for agent interactions."""
    def __init__(self):
        """Initialize BasePrompt."""
        logger.info("BasePrompt initialized")

    def generate_prompt(self, request: PromptRequest) -> str:
        """Generate a prompt based on context and task type.

        Args:
            request (PromptRequest): Prompt generation request.

        Returns:
            str: Generated prompt.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            prompt_templates = {
                "translator": "Translate the following text: {context}",
                "library": "Summarize the following document: {context}"
            }
            template = prompt_templates.get(request.task_type, "Process the following: {context}")
            prompt = template.format(context=request.context)
            logger.info(f"Generated prompt for task {request.task_type}")
            return prompt
        except Exception as e:
            logger.error(f"Failed to generate prompt: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [BasePrompt] Failed to generate prompt: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Failed to generate prompt: {str(e)}")
