# main/server/mcp/utils/base_prompt.py
from typing import Dict, Any, Optional
from ..utils.mcp_error_handler import MCPError
import logging
import os

logger = logging.getLogger("mcp")

class BasePrompt:
    def __init__(self):
        self.default_prompt = os.getenv("DEFAULT_PROMPT", "You are a helpful AI assistant for the Vial MCP Controller. Provide accurate and concise responses.")

    def generate_prompt(self, task: str, context: Dict[str, Any], user_id: str) -> str:
        try:
            if not task or not user_id:
                raise MCPError(code=-32602, message="Task and user ID are required")
            
            # Base prompt structure
            prompt = f"{self.default_prompt}\n\nTask: {task}\n"
            
            # Add context if provided
            if context:
                prompt += "Context:\n"
                for key, value in context.items():
                    prompt += f"{key}: {value}\n"
            
            prompt += f"User: {user_id}\nRespond in a professional and secure manner."
            
            logger.debug(f"Generated prompt for user {user_id}: {prompt}")
            return prompt
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to generate prompt: {str(e)}")

    def validate_prompt(self, prompt: str) -> bool:
        try:
            if not prompt or len(prompt) > 5000:
                raise MCPError(code=-32602, message="Prompt is empty or exceeds 5000 characters")
            if any(word in prompt.lower() for word in ["malicious", "hack", "exploit"]):
                raise MCPError(code=-32602, message="Prompt contains prohibited terms")
            return True
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Prompt validation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to validate prompt: {str(e)}")
