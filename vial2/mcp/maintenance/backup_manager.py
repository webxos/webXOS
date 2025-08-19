from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import json
import shutil
import os

logger = logging.getLogger(__name__)

class BackupManager:
    def create_backup(self):
        try:
            backup_dir = "backups/vial2_" + time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backups"))
            logger.info(f"Created backup at {backup_dir}")
        except Exception as e:
            error_logger.log_error("backup_create", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Backup creation failed: {str(e)}")
            raise

backup_manager = BackupManager()

# xAI Artifact Tags: #vial2 #mcp #maintenance #backup #neon_mcp
