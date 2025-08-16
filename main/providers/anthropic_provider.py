import anthropic
from .base_provider import BaseLLMProvider
from typing import Dict, Any, List

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            response = await self.client.completions.create(
                model=kwargs.get("model", "claude-3-sonnet"),
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return {
                "provider": "anthropic",
                "model": response.model,
                "response": response.completion,
                "usage": response.usage.dict()
            }
        except Exception as e:
            return {"error": str(e), "provider": "anthropic"}
    
    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        try:
            response = await self.client.messages.create(
                model=kwargs.get("model", "claude-3-sonnet"),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return {
                "provider": "anthropic",
                "model": response.model,
                "response": response.content[0].text,
                "usage": response.usage.dict()
            }
        except Exception as e:
            return {"error": str(e), "provider": "anthropic"}
    
    async def embedding(self, text: str) -> List[float]:
        return []
