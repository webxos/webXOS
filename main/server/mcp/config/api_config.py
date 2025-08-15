# main/server/mcp/config/api_config.py
from typing import Dict, List
from pydantic import BaseModel

class EndpointConfig(BaseModel):
    path: str
    method: str
    description: str
    requires_auth: bool

class APIConfig:
    def __init__(self):
        self.endpoints = self._load_endpoints()
        self.settings = {
            "jsonrpc_version": "2.0",
            "mcp_version": "1.0",
            "max_request_size": 1024 * 1024,  # 1MB
            "rate_limit": {
                "requests_per_minute": 60,
                "burst_size": 10
            }
        }

    def _load_endpoints(self) -> List[EndpointConfig]:
        return [
            EndpointConfig(
                path="/mcp",
                method="POST",
                description="Main MCP endpoint for JSON-RPC 2.0 requests",
                requires_auth=True
            ),
            EndpointConfig(
                path="/auth/login",
                method="POST",
                description="Authenticate user via OAuth 3.0 or WebAuthn",
                requires_auth=False
            ),
            EndpointConfig(
                path="/health",
                method="GET",
                description="Health check endpoint",
                requires_auth=False
            ),
            EndpointConfig(
                path="/ready",
                method="GET",
                description="Readiness check endpoint",
                requires_auth=False
            )
        ]

    def get_endpoint(self, path: str, method: str) -> Dict[str, Any]:
        endpoint = next((e for e in self.endpoints if e.path == path and e.method == method), None)
        if not endpoint:
            raise ValueError(f"Endpoint not found: {method} {path}")
        return endpoint.dict()

    def get_settings(self) -> Dict[str, Any]:
        return self.settings
