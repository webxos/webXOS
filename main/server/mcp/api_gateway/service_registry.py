# server/mcp/api_gateway/service_registry.py
from typing import Dict, Optional
from ..utils.error_handler import handle_service_error

class ServiceRegistry:
    def __init__(self):
        self.services: Dict[str, Dict] = {}

    def register_service(self, service_name: str, endpoint: str, metadata: Optional[Dict] = None):
        try:
            self.services[service_name] = {
                "endpoint": endpoint,
                "metadata": metadata or {},
                "status": "active"
            }
        except Exception as e:
            handle_service_error(e)
            raise

    def deregister_service(self, service_name: str):
        if service_name in self.services:
            del self.services[service_name]

    def get_service(self, service_name: str) -> Optional[Dict]:
        return self.services.get(service_name)

    def list_services(self) -> Dict:
        return self.services
