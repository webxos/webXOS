import json
import os
from typing import Dict, Optional
from ...error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = "vial2/config/config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            default_config = {"api_port": 8000, "neon_db_uri": "placeholder", "oauth_client_id": "placeholder"}
            self._save_config(default_config)
            return default_config
        except json.JSONDecodeError as e:
            error_logger.log_error("config_load_json", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Failed to decode config: {str(e)}")
            raise
        except Exception as e:
            error_logger.log_error("config_load", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Config load failed: {str(e)}")
            raise

    def _save_config(self, config: Dict) -> None:
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            error_logger.log_error("config_save", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=config)
            logger.error(f"Config save failed: {str(e)}")
            raise

    def get(self, key: str) -> Optional[str]:
        return self.config.get(key)

    def set(self, key: str, value: str) -> None:
        try:
            self.config[key] = value
            self._save_config(self.config)
        except Exception as e:
            error_logger.log_error("config_set", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={key: value})
            logger.error(f"Config set failed: {str(e)}")
            raise

config = ConfigManager()

# xAI Artifact Tags: #vial2 #mcp #config #manager #neon_mcp
