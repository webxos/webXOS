from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import os
import shutil

logger = logging.getLogger(__name__)

class BackupManager:
    async def create_backup(self, vial_id: str):
        try:
            backup_path = os.path.join(os.getenv("BACKUP_PATH", "/app/backups"), f"{vial_id}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
            query = "SELECT event_data FROM vial_logs WHERE vial_id = $1"
            data = await neon_db.execute(query, vial_id)
            os.makedirs(backup_path, exist_ok=True)
            with open(os.path.join(backup_path, "backup.json"), "w") as f:
                json.dump(data, f)
            logger.info(f"Created backup for vial {vial_id} at {backup_path}")
        except Exception as e:
            error_logger.log_error("backup_create", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Backup creation failed: {str(e)}")
            raise

backup_manager = BackupManager()

# xAI Artifact Tags: #vial2 #mcp #backup #manager #neon #neon_mcp
