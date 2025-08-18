import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_jsonrpc_request():
    try:
        response = client.post("/mcp/api/jsonrpc", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/list",
            "params": {}
        })
        assert response.status_code == 200
        assert response.json()["jsonrpc"] == "2.0"
        assert response.json()["id"] == 1
        assert "result" in response.json()
    except Exception as e:
        error_logger.log_error("test_jsonrpc_request", str(e), str(e.__traceback__))
        logger.error(f"JSON-RPC request test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_jsonrpc_notification():
    try:
        response = client.post("/mcp/api/jsonrpc", json={
            "jsonrpc": "2.0",
            "method": "prompts/list",
            "params": {}
        })
        assert response.status_code == 200
        assert response.json() is None
    except Exception as e:
        error_logger.log_error("test_jsonrpc_notification", str(e), str(e.__traceback__))
        logger.error(f"JSON-RPC notification test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_jsonrpc_invalid_method():
    try:
        response = client.post("/mcp/api/jsonrpc", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "invalid_method",
            "params": {}
        })
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32601
    except Exception as e:
        error_logger.log_error("test_jsonrpc_invalid_method", str(e), str(e.__traceback__))
        logger.error(f"JSON-RPC invalid method test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #jsonrpc #neon_mcp
