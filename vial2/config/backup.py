import json
import os
from ..error_logging.error_log import error_logger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def backup_config():
    try:
        config_data = {
            "DATABASE_URL": os.getenv("DATABASE_URL"),
            "STACK_AUTH_CLIENT_ID": os.getenv("STACK_AUTH_CLIENT_ID"),
            "STACK_AUTH_CLIENT_SECRET": os.getenv("STACK_AUTH_CLIENT_SECRET"),
            "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY"),
            "REPO_PATH": os.getenv("REPO_PATH"),
            "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "").split(","),
            "backup_timestamp": datetime.utcnow().isoformat()
        }
        backup_file = f"config/backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        with open(backup_file, "w") as f:
            json.dump(config_data, f, indent=2)
        return {"status": "success", "backup_file": backup_file}
    except Exception as e:
        error_logger.log_error("config_backup", f"Config backup failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Config backup failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #config #backup #neon_mcp
