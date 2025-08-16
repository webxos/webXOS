from .base_provider import BaseLLMProvider
from typing import Dict, Any, List

class GoogleProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {"error": "Not implemented", "provider": "google"}
    
    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        return {"error": "Not implemented", "provider": "google"}
    
    async def embedding(self, text: str) -> List[float]:
        return []
