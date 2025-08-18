import pytest
from ..task_manager import assign_task
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_task_assignment():
    try:
        task_data = {"task_id": "task123", "description": "Test task"}
        result = await assign_task("vial1", task_data)
        assert result["status"] == "success"
        assert result["task_id"] == "task123"
    except Exception as e:
        error_logger.log_error("test_task_manager", str(e), str(e.__traceback__))
        logger.error(f"Task manager test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #task_manager #neon_mcp
