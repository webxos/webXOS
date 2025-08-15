# main/server/mcp/agents/translator_agent.py
from typing import Dict, Any, Optional
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
import logging
import os
import aiohttp
import json

logger = logging.getLogger("mcp")

class TranslatorAgent:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.api_key = os.getenv("TRANSLATION_API_KEY", "")
        self.api_url = os.getenv("TRANSLATION_API_URL", "https://api.example.com/translate")
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh"]

    @self.metrics.track_request("translate_config")
    async def translate_config(self, config: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
        try:
            if not config or not target_lang:
                raise MCPError(code=-32602, message="Config and target language are required")
            if target_lang not in self.supported_languages:
                raise MCPError(code=-32602, message=f"Unsupported language: {target_lang}")
            
            translated_config = config.copy()
            for key, value in config.items():
                if isinstance(value, str) and value:
                    translated_config[key] = await self._translate_text(value, target_lang)
                elif isinstance(value, dict):
                    translated_config[key] = await self.translate_config(value, target_lang)
            
            logger.info(f"Translated config to {target_lang}")
            return translated_config
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Config translation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to translate config: {str(e)}")

    async def _translate_text(self, text: str, target_lang: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"text": text, "target_lang": target_lang}
                ) as response:
                    if response.status != 200:
                        raise MCPError(code=-32603, message=f"Translation API error: {response.status}")
                    data = await response.json()
                    return data.get("translated_text", text)
        except Exception as e:
            logger.error(f"Text translation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to translate text: {str(e)}")

    @self.metrics.track_request("translate_prompt")
    async def translate_prompt(self, prompt: str, target_lang: str, user_id: str) -> str:
        try:
            if not prompt or not target_lang or not user_id:
                raise MCPError(code=-32602, message="Prompt, target language, and user ID are required")
            if target_lang not in self.supported_languages:
                raise MCPError(code=-32602, message=f"Unsupported language: {target_lang}")
            
            translated = await self._translate_text(prompt, target_lang)
            logger.info(f"Translated prompt for user {user_id} to {target_lang}")
            return translated
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Prompt translation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to translate prompt: {str(e)}")

    def close(self):
        pass  # No resources to close in this implementation
