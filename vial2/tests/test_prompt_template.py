import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_prompt_template():
    try:
        response = client.post("/mcp/api/vial/prompt/template", json={"vial_id": "vial1", "prompt_name": "test_prompt", "template": {"text": "Sample prompt"}}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["data"][0]["template"]["text"] == "Sample prompt"
    except Exception as e:
        error_logger.log_error("test_prompt_template", str(e), str(e.__traceback__), sql_statement="INSERT INTO prompt_templates", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Prompt template test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_prompt_template_missing_vial():
    try:
        response = client.post("/mcp/api/vial/prompt/template", json={"prompt_name": "test_prompt"}, headers={"Authorization": "Bearer test_token"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32602
    except Exception as e:
        error_logger.log_error("test_prompt_template_missing_vial", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
        logger.error(f"Prompt template missing vial test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #prompt_template #sqlite #octokit #neon_mcp
