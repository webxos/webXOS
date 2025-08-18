import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_pytorch_optimizer():
    try:
        response = client.post("/mcp/api/vial/optimizer/config", json={"vial_id": "vial1", "optimizer_type": "Adam", "learning_rate": 0.001}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["optimizer_config"]["type"] == "Adam"
    except Exception as e:
        error_logger.log_error("test_pytorch_optimizer", str(e), str(e.__traceback__), sql_statement="UPDATE vials SET optimizer_config", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"PyTorch optimizer test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_pytorch_optimizer_missing_vial():
    try:
        response = client.post("/mcp/api/vial/optimizer/config", json={"optimizer_type": "Adam"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_pytorch_optimizer_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"PyTorch optimizer missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #pytorch #optimizer #sqlite #octokit #neon_mcp
