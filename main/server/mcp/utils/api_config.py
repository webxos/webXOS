# main/server/mcp/utils/api_config.py
from typing import Dict, Any, List
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
import os
import logging
import json

logger = logging.getLogger("mcp")

class APIConfig:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.config_file = os.getenv("API_CONFIG_FILE", "api_config.json")
        self.default_config = {
            "endpoints": {
                "mcp.createSession": {"enabled": True, "auth_required": True},
                "mcp.initiateMFA": {"enabled": True, "auth_required": False},
                "mcp.verifyMFA": {"enabled": True, "auth_required": False},
                "mcp.createNote": {"enabled": True, "auth_required": True},
                "mcp.addSubNote": {"enabled": True, "auth_required": True},
                "mcp.simulateCircuit": {"enabled": True, "auth_required": True},
                "mcp.getSystemMetrics": {"enabled": True, "auth_required": True},
                "mcp.createAgent": {"enabled": True, "auth_required": True},
                "mcp.executeWorkflow": {"enabled": True, "auth_required": True},
                "mcp.addSubIssue": {"enabled": True, "auth_required": True},
                "mcp.subscribe": {"enabled": True, "auth_required": True},
                "mcp.publish": {"enabled": True, "auth_required": True}
            },
            "cors": {
                "allowed_origins": os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
                "allow_credentials": True,
                "allow_methods": ["POST", "GET"],
                "allow_headers": ["Authorization", "Content-Type"]
            }
        }

    def load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    logger.info("Loaded API configuration from file")
                    return config
            logger.info("Using default API configuration")
            return self.default_config
        except Exception as e:
            logger.error(f"Failed to load API config: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to load API config: {str(e)}")

    def validate_endpoint(self, endpoint: str) -> bool:
        try:
            config = self.load_config()
            return config["endpoints"].get(endpoint, {}).get("enabled", False)
        except Exception as e:
            logger.error(f"Failed to validate endpoint {endpoint}: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to validate endpoint: {str(e)}")

    def get_endpoint_config(self, endpoint: str) -> Dict[str, Any]:
        try:
            config = self.load_config()
            return config["endpoints"].get(endpoint, {})
        except Exception as e:
            logger.error(f"Failed to get endpoint config for {endpoint}: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to get endpoint config: {str(e)}")

    def get_cors_config(self) -> Dict[str, Any]:
        try:
            config = self.load_config()
            return config["cors"]
        except Exception as e:
            logger.error(f"Failed to get CORS config: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to get CORS config: {str(e)}")
