import httpx
from .base_provider import BaseLLMProvider
from typing import Dict, Any, List

class XAIProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.x.ai/v1")
        
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": kwargs.get("model", "grok-1"),
                        "prompt": prompt,
                        "max_tokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "provider": "xai",
                        "model": data.get("model"),
                        "response": data["choices"][0]["text"],
                        "usage": data.get("usage", {})
                    }
                else:
                    return {"error": f"HTTP {response.status_code}", "provider": "xai"}
        except Exception as e:
            return {"error": str(e), "provider": "xai"}
    
    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": kwargs.get("model", "grok-1"),
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "provider": "xai",
                        "model": data.get("model"),
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {})
                    }
                else:
                    return {"error": f"HTTP {response.status_code}", "provider": "xai"}
        except Exception as e:
            return {"error": str(e), "provider": "xai"}
    
    async def embedding(self, text: str) -> List[float]:
        return []
