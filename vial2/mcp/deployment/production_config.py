import os
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ProductionConfig:
    def __init__(self):
        self.config = {
            "host": os.getenv("PROD_HOST", "0.0.0.0"),
            "port": int(os.getenv("PROD_PORT", "8000")),
            "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true",
            "backup_path": os.getenv("BACKUP_PATH", "/app/backups")
        }

    def apply_config(self):
        try:
            logger.info(f"Applied production config: {self.config}")
            os.makedirs(self.config["backup_path"], exist_ok=True)
        except Exception as e:
            error_logger.log_error("prod_config_apply", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=self.config)
            logger.error(f"Production config application failed: {str(e)}")
            raise

prod_config = ProductionConfig()

# xAI Artifact Tags: #vial2 #mcp #deployment #production #neon_mcp
