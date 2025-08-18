import pytest
from ..utils.telemetry import Telemetry
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_telemetry_collection():
    try:
        telemetry = Telemetry()
        telemetry.record_event("test_event", {"value": 42})
        assert telemetry.get_metrics()["test_event"] == 1
    except Exception as e:
        error_logger.log_error("test_telemetry", str(e), str(e.__traceback__))
        logger.error(f"Telemetry test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #telemetry #neon_mcp
