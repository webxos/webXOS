# main/server/mcp/utils/api_config.py
from typing import Dict, List
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from pydantic import BaseModel
import os

class EndpointConfig(BaseModel):
    name: str
    url: str
    method: str
    requires_auth: bool = True

class APIConfig:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.load_config()

    def load_config(self) -> None:
        with self.metrics.track_span("load_api_config"):
            try:
                default_endpoints = [
                    EndpointConfig(name="auth", url=os.getenv("AUTH_SERVICE_URL", "http://localhost:8002"), method="POST"),
                    EndpointConfig(name="quantum", url=os.getenv("QUANTUM_SERVICE_URL", "http://localhost:8001"), method="POST"),
                    EndpointConfig(name="wallet", url=os.getenv("WALLET_SERVICE_URL", "http://localhost:8000/wallet"), method="GET"),
                    EndpointConfig(name="notes", url=os.getenv("NOTES_SERVICE_URL", "http://localhost:8000/notes"), method="GET"),
                    EndpointConfig(name="vials", url=os.getenv("VIALS_SERVICE_URL", "http://localhost:8000/vials"), method="GET"),
                    EndpointConfig(name="ai", url=os.getenv("AI_SERVICE_URL", "http://localhost:8000/ai"), method="POST")
                ]
                for endpoint in default_endpoints:
                    self.endpoints[endpoint.name] = endpoint
            except Exception as e:
                handle_generic_error(e, context="load_api_config")
                raise

    def get_endpoint(self, name: str) -> EndpointConfig:
        with self.metrics.track_span("get_endpoint", {"name": name}):
            try:
                endpoint = self.endpoints.get(name)
                if not endpoint:
                    raise ValueError(f"Endpoint {name} not found")
                return endpoint
            except Exception as e:
                handle_generic_error(e, context="get_endpoint")
                raise

    def list_endpoints(self) -> List[EndpointConfig]:
        with self.metrics.track_span("list_endpoints"):
            try:
                return list(self.endpoints.values())
            except Exception as e:
                handle_generic_error(e, context="list_endpoints")
                raise
