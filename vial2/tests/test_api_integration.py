import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_auth_relay_check():
    try:
        response = client.get("/mcp/auth/relay_check")
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_auth_relay", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Auth relay test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_console_command():
    try:
        response = client.post("/mcp/api/vial/console/execute", json={"command": "/mcp status", "vial_id": "vial1"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
    except Exception as e:
        error_logger.log_error("test_console_command", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Console command test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #api #integration #neon_mcp
