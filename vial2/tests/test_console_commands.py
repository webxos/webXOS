import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_console_command_execution():
    try:
        response = client.post("/mcp/api/vial/console/execute", json={"command": "/mcp status", "vial_id": "vial1"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert "connected_resources" in response.json()["result"]["data"]
    except Exception as e:
        error_logger.log_error("test_console_command", str(e), str(e.__traceback__), sql_statement="INSERT INTO vial_logs", sql_error_code=None, params={"command": "/mcp status"})
        logger.error(f"Console command test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_console_command_invalid():
    try:
        response = client.post("/mcp/api/vial/console/execute", json={"command": "/exec malicious"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_console_command_invalid", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={"command": "/exec malicious"})
        logger.error(f"Invalid console command test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #console #commands #sqlite #neon_mcp
