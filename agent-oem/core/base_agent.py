from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a given action with parameters."""
        pass
