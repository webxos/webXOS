import pytest
from mcp.maintenance.backup_manager import backup_manager
import logging
import os
import shutil

logger = logging.getLogger(__name__)

def test_backup_creation():
    try:
        backup_manager.create_backup()
        assert os.path.exists("backups")
        shutil.rmtree("backups", ignore_errors=True)
        logger.info("Backup test passed")
    except Exception as e:
        logger.error(f"Backup test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #backup #neon_mcp
