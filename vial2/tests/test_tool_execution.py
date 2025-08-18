import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_tool_execution():
    try:
        response = client.post("/mcp/api/vial/tool/execute", json={"vial_id": "vial1", "tool_name": "test_tool", "args": {"param": "value"}}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"]["tool_name"] == "test_tool"
    except Exception as e:
        error_logger.log_error("test_tool_execution", str(e), str(e.__traceback__), sql_statement="SELECT tool_config FROM mcp_tools", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Tool execution test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_tool_execution_missing_vial():
    try:
        response = client.post("/mcp/api/vial/tool/execute", json={"tool_name": "test_tool"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_tool_execution_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Tool execution missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #tool_execution #sqlite #octokit #neon_mcp
