# main/server/mcp/utils/base_prompt.py
from typing import Dict, Any, Optional
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
from ..agents.translator_agent import TranslatorAgent
import logging
import json

logger = logging.getLogger("mcp")

class BasePrompt:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.translator = TranslatorAgent()
        self.default_prompt = {
            "system": "You are an AI assistant for the Vial MCP Controller, designed to manage resources, execute workflows, and provide secure, efficient responses. Follow JSON-RPC 2.0 standards and maintain security best practices.",
            "user": "Provide a response to the following request: {{request}}",
            "constraints": {
                "max_tokens": 1000,
                "language": "en",
                "response_format": "json"
            }
        }

    @self.metrics.track_request("generate_prompt")
    async def generate_prompt(self, request: Dict[str, Any], user_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not request or not user_id:
                raise MCPError(code=-32602, message="Request and user ID are required")
            
            prompt = self.default_prompt.copy()
            prompt["user"] = prompt["user"].replace("{{request}}", json.dumps(request))
            
            if language and language != "en":
                prompt = await self.translator.translate_config(prompt, language)
            
            logger.info(f"Generated prompt for user {user_id}, language: {language or 'en'}")
            return prompt
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to generate prompt: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to generate prompt: {str(e)}")

    @self.metrics.track_request("validate_prompt")
    async def validate_prompt(self, prompt: Dict[str, Any]) -> bool:
        try:
            required_keys = ["system", "user", "constraints"]
            if not all(key in prompt for key in required_keys):
                raise MCPError(code=-32602, message="Prompt missing required keys")
            if not isinstance(prompt["constraints"].get("max_tokens"), int):
                raise MCPError(code=-32602, message="Invalid max_tokens in constraints")
            if prompt["constraints"].get("response_format") != "json":
                raise MCPError(code=-32602, message="Response format must be JSON")
            logger.info("Prompt validated successfully")
            return True
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to validate prompt: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to validate prompt: {str(e)}")

    async def close(self):
        self.translator.close()
