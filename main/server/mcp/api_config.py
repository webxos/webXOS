import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class APIConfig:
    """Centralizes API configuration settings for Vial MCP."""
    def __init__(self):
        """Initialize APIConfig with environment-based settings."""
        self.api_settings = {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("PYTHON_DOCKER_PORT", 8000)),
            "log_level": os.getenv("LOG_LEVEL", "info"),
            "allowed_origins": os.getenv("ALLOWED_ORIGINS", "https://webxos.netlify.app").split(","),
            "max_request_size": int(os.getenv("MAX_REQUEST_SIZE", 1000000)),  # 1MB
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
            "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", 60))  # Seconds
        }
        logger.info("APIConfig initialized")

    def get_config(self) -> Dict:
        """Get API configuration settings.

        Returns:
            Dict: Configuration settings.
        """
        return self.api_settings

    def update_config(self, key: str, value: any) -> None:
        """Update a specific configuration setting.

        Args:
            key (str): Configuration key to update.
            value (any): New value for the key.

        Raises:
            HTTPException: If key is invalid.
        """
        try:
            if key not in self.api_settings:
                error_msg = f"Invalid config key: {key}"
                logger.error(error_msg)
                with open("/app/errorlog.md", "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] [APIConfig] {error_msg}\n")
                raise HTTPException(status_code=400, detail=error_msg)
            self.api_settings[key] = value
            logger.info(f"Updated config: {key} = {value}")
        except Exception as e:
            logger.error(f"Config update failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [APIConfig] Config update failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")
