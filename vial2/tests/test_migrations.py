import pytest
from ..migrations import run_migrations
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_database_migrations():
    try:
        result = await run_migrations()
        assert result["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_migrations", str(e), str(e.__traceback__))
        logger.error(f"Migration test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #migrations #neon_mcp
