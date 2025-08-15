# main/server/mcp/api_gateway/service_registry.py
from typing import Dict, Any, Callable, Optional
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.api_config import APIConfig
import logging
import asyncio

logger = logging.getLogger("mcp")

class ServiceRegistry:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.api_config = APIConfig()
        self.services: Dict[str, Callable] = {}
    
    def register_service(self, method: str, handler: Callable) -> None:
        try:
            if not method or not handler:
                raise MCPError(code=-32602, message="Method and handler are required")
            if not self.api_config.validate_endpoint(method):
                raise MCPError(code=-32601, message=f"Method {method} is not enabled")
            self.services[method] = handler
            logger.info(f"Registered service: {method}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to register service {method}: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to register service: {str(e)}")

    async def dispatch(self, method: str, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        try:
            if method not in self.services:
                raise MCPError(code=-32601, message=f"Method {method} not found")
            
            self.metrics.requests_total.labels(endpoint=method).inc()
            result = await self.services[method](params)
            
            logger.debug(f"Dispatched {method} with params: {params}")
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except MCPError as e:
            logger.error(f"MCPError in dispatch {method}: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": e.code, "message": str(e), "data": e.data},
                "id": request_id
            }
        except Exception as e:
            logger.error(f"Unexpected error in dispatch {method}: {str(e)}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {str(e)}", "data": {"traceback": str(e)}},
                "id": request_id
            }

    def get_registered_services(self) -> Dict[str, Any]:
        return {method: {"enabled": self.api_config.validate_endpoint(method)} for method in self.services.keys()}
