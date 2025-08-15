# main/server/mcp/utils/config_validator.py
from typing import Dict, Any, List
import os
import logging
import re
from ..utils.mcp_error_handler import MCPError

logger = logging.getLogger("mcp")

class ConfigValidator:
    def __init__(self):
        self.required_configs = [
            "MCP_SERVER_HOST",
            "MCP_SERVER_PORT",
            "MONGODB_URI",
            "REDIS_URI",
            "SECRET_KEY",
            "JWT_ALGORITHM",
            "ALLOWED_ORIGINS"
        ]
        self.token_pattern = re.compile(r"^[a-zA-Z0-9\-_\.]{32,}$")  # Basic token validation

    def validate_config(self, config: Dict[str, Any]) -> None:
        try:
            # Check required configurations
            for key in self.required_configs:
                if key not in config or not config[key]:
                    raise MCPError(code=-32602, message=f"Missing or empty configuration: {key}")
            
            # Validate server host and port
            if not isinstance(config["MCP_SERVER_PORT"], (int, str)) or int(config["MCP_SERVER_PORT"]) <= 0:
                raise MCPError(code=-32602, message="Invalid MCP_SERVER_PORT")
            
            # Validate MongoDB and Redis URIs
            for uri_key in ["MONGODB_URI", "REDIS_URI"]:
                if not config[uri_key].startswith(("mongodb://", "redis://")):
                    raise MCPError(code=-32602, message=f"Invalid {uri_key} format")
            
            # Validate secret key strength
            if len(config["SECRET_KEY"]) < 32:
                raise MCPError(code=-32602, message="SECRET_KEY must be at least 32 characters")
            
            # Validate allowed origins
            if not isinstance(config["ALLOWED_ORIGINS"], (str, list)):
                raise MCPError(code=-32602, message="ALLOWED_ORIGINS must be a string or list")
            
            logger.info("Configuration validated successfully")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to validate configuration: {str(e)}")

    def validate_token(self, token: str, scope: str) -> bool:
        try:
            if not self.token_pattern.match(token):
                raise MCPError(code=-32602, message="Invalid token format")
            
            # Check for least-privilege scope (inspired by GitHub MCP recommendations)
            if scope == "repo:all" and os.getenv("ENVIRONMENT") == "production":
                logger.warning("Broad scope 'repo:all' detected in production")
                raise MCPError(code=-32602, message="Broad scope 'repo:all' not allowed in production")
            
            return True
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to validate token: {str(e)}")

    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sanitized = config.copy()
            sensitive_keys = ["SECRET_KEY", "TRANSLATION_API_KEY", "WEB3_PROVIDER_URL"]
            for key in sensitive_keys:
                if key in sanitized:
                    sanitized[key] = "****"
            return sanitized
        except Exception as e:
            logger.error(f"Config sanitization failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to sanitize configuration: {str(e)}")
