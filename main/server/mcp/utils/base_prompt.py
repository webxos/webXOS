# main/server/mcp/utils/base_prompt.py
from typing import Dict, Any
import logging

logger = logging.getLogger("mcp")

class BasePrompt:
    async def generate_prompt(self, params: Dict[str, Any], user_id: str, lang: str = "en") -> str:
        try:
            # Mock prompt generation
            return f"Prompt for {user_id} in {lang}: {json.dumps(params)}"
        except Exception as e:
            logger.error(f"Prompt generation error: {str(e)}", exc_info=True)
            raise ValueError(f"Prompt generation failed: {str(e)}")
