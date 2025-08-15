# main/server/mcp/utils/base_prompt.py
from typing import Dict, Optional
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error

class BasePrompt:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.default_system_prompt = (
            "You are Grok, created by xAI. You are a highly capable AI assistant designed to provide accurate, helpful, and context-aware responses. "
            "Your responses should be clear, concise, and aligned with the user's intent. Use the provided context to tailor your answers, "
            "and ensure all responses adhere to xAI's mission of advancing human scientific discovery."
        )

    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[Dict] = None) -> Dict:
        with self.metrics.track_span("format_prompt"):
            try:
                prompt = {
                    "system": system_prompt or self.default_system_prompt,
                    "user": user_input
                }
                if context:
                    prompt["context"] = context
                return prompt
            except Exception as e:
                handle_generic_error(e, context="format_prompt")
                raise

    def validate_prompt(self, prompt: Dict) -> bool:
        with self.metrics.track_span("validate_prompt"):
            try:
                if not isinstance(prompt, dict) or "user" not in prompt or not prompt["user"]:
                    return False
                if "system" not in prompt:
                    prompt["system"] = self.default_system_prompt
                return True
            except Exception as e:
                handle_generic_error(e, context="validate_prompt")
                return False
