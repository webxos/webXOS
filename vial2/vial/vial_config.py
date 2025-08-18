from ..config import Config
from ..error_logging.error_log import error_logger
import logging
import json

logger = logging.getLogger(__name__)

class VialConfig:
    def __init__(self):
        self.config = {
            "vial_max_concurrent": Config.VIAL_MAX_CONCURRENT or 10,
            "vial_timeout": Config.VIAL_TIMEOUT or 300,
            "quantum_enabled": Config.QUANTUM_ENABLED or True
        }

    def update_config(self, new_config: dict):
        try:
            self.config.update(new_config)
            with open("vial_config.json", "w") as f:
                json.dump(self.config, f)
            logger.info("Vial configuration updated")
            return {"status": "success", "config": self.config}
        except Exception as e:
            error_logger.log_error("vial_config_update", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=new_config)
            logger.error(f"Vial config update failed: {str(e)}")
            raise

    def get_config(self):
        try:
            return {"status": "success", "config": self.config}
        except Exception as e:
            error_logger.log_error("vial_config_get", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
            logger.error(f"Vial config retrieval failed: {str(e)}")
            raise

vial_config = VialConfig()

# xAI Artifact Tags: #vial2 #vial #config #sqlite #neon_mcp
