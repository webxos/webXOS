import pytest
import asyncpg
from ..utils.helpers import get_db_pool, log_event
from ..config import config
from ..error_logging.error_log import error_logger
from datetime import datetime

@pytest.mark.asyncio
async def test_get_db_pool():
    try:
        async with get_db_pool() as conn:
            result = await conn.fetch("SELECT 1")
            assert result[0][0] == 1
    except Exception as e:
        error_logger.log_error("test_utils", f"Test get_db_pool failed: {str(e)}", str(e.__traceback__))
        raise

@pytest.mark.asyncio
async def test_log_event():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        await log_event("test_event", "Test event message", db)
        result = await db.fetch("SELECT * FROM logs WHERE event_type=$1", "test_event")
        assert len(result) > 0
        assert result[0]["message"] == "Test event message"
    except Exception as e:
        error_logger.log_error("test_utils", f"Test log_event failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #tests #utils #neon_mcp
