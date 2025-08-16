from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "default")
        
    @abstractmethod
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def embedding(self, text: str) -> List[float]:
        pass
