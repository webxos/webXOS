import pytest
import asyncpg
from ..agents import handle_command, configure_compute, refresh_configuration, terminate_fast, terminate_immediate
from ..config import config
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_handle_command():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        result = await handle_command("/prompt vial1 train model", {"command": "/prompt vial1 train model"}, db)
        assert result["status"] == "success"
        assert result["vial_id"] == "vial1"
        assert result["action"] == "train model"
    except Exception as e:
        error_logger.log_error("test_agents", f"Test handle_command failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_configure_compute():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        spec = {"model": "resnet", "epochs": 10}
        result = await configure_compute(spec, db)
        assert result["status"] == "success"
        assert result["spec"] == spec
    except Exception as e:
        error_logger.log_error("test_agents", f"Test configure_compute failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_refresh_configuration():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        result = await refresh_configuration(db)
        assert result["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_agents", f"Test refresh_configuration failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_terminate_fast():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        result = await terminate_fast(db)
        assert result["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_agents", f"Test terminate_fast failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_terminate_immediate():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        result = await terminate_immediate(db)
        assert result["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_agents", f"Test terminate_immediate failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #tests #agents #neon_mcp
