import logging
from fastapi import HTTPException
from pydantic import BaseModel
from ..db.db_manager import DatabaseManager
from ..utils.base_prompt import BasePrompt
from ..error_handler import ErrorHandler
import aiohttp
import os

logger = logging.getLogger(__name__)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslatorAgent:
    """Handles text translation tasks for Vial MCP."""
    def __init__(self, db_manager: DatabaseManager = None, error_handler: ErrorHandler = None):
        """Initialize TranslatorAgent with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.base_prompt = BasePrompt()
        self.error_handler = error_handler or ErrorHandler()
        self.api_key = os.getenv("GROK_API_KEY")
        logger.info("TranslatorAgent initialized")

    async def process_task(self, parameters: dict) -> dict:
        """Process a translation task.

        Args:
            parameters (dict): Translation request parameters (text, source_lang, target_lang).

        Returns:
            dict: Translation result.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            request = TranslationRequest(**parameters)
            prompt = self.base_prompt.generate_prompt({"context": request.text, "task_type": "translator"})
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/grok/translate",
                    json={"prompt": prompt, "source_lang": request.source_lang, "target_lang": request.target_lang},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                ) as response:
                    if response.status != 200:
                        error_msg = f"Translation API failed: {response.status}"
                        logger.error(error_msg)
                        self.error_handler.handle_exception("/api/agents/translator", "translator", Exception(error_msg))
                    result = await response.json()
                    translation = result.get("translated_text", "")
                    await self.db_manager.log_translation("translator", request.text, translation, request.source_lang, request.target_lang)
                    logger.info(f"Translated text from {request.source_lang} to {request.target_lang}")
                    return {"translated_text": translation, "source_lang": request.source_lang, "target_lang": request.target_lang}
        except Exception as e:
            self.error_handler.handle_exception("/api/agents/translator", "translator", e)
