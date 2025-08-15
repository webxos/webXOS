# main/server/mcp/api_gateway/service_registry.py
from typing import Dict, Callable, Any
import logging

logger = logging.getLogger("mcp")

class ServiceRegistry:
    def __init__(self):
        self.services: Dict[str, Callable] = {}

    def register_service(self, method: str, handler: Callable) -> None:
        if method in self.services:
            logger.warning(f"Overwriting service: {method}")
        self.services[method] = handler
        logger.info(f"Registered service: {method}")

    async def dispatch(self, method: str, params: Dict[str, Any], request_id: int = None) -> Dict[str, Any]:
        try:
            if method not in self.services:
                raise ValueError(f"Method {method} not found")
            result = await self.services[method](params)
            return {"jsonrpc": "2.0", "result": result, "id": request_id} if request_id else {"jsonrpc": "2.0", "result": result}
        except Exception as e:
            logger.error(f"Dispatch error for {method}: {str(e)}", exc_info=True)
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": str(e)}, "id": request_id}
