import openai
from .base_provider import BaseLLMProvider
from typing import Dict, Any, List

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            response = await self.client.completions.create(
                model=kwargs.get("model", "gpt-4"),
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return {
                "provider": "openai",
                "model": response.model,
                "response": response.choices[0].text,
                "usage": response.usage.dict()
            }
        except Exception as e:
            return {"error": str(e), "provider": "openai"}
    
    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", "gpt-4"),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return {
                "provider": "openai",
                "model": response.model,
                "response": response.choices[0].message.content,
                "usage": response.usage.dict()
            }
        except Exception as e:
            return {"error": str(e), "provider": "openai"}
    
    async def embedding(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            return []
