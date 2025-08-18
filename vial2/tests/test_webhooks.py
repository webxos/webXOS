import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger

client = TestClient(app)

@pytest.mark.asyncio
async def test_handle_webhook():
    try:
        response = client.post("/mcp/api/webhooks", json={"event_type": "test_event"})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["event_type"] == "test_event"
    except Exception as e:
        error_logger.log_error("test_webhooks", f"Test handle_webhook failed: {str(e)}", str(e.__traceback__))
        raise

@pytest.mark.asyncio
async def test_webhook_missing_event_type():
    try:
        response = client.post("/mcp/api/webhooks", json={})
        assert response.status_code == 400
        assert "Missing event_type" in response.json()["detail"]
    except Exception as e:
        error_logger.log_error("test_webhooks", f"Test webhook missing event type failed: {str(e)}", str(e.__traceback__))
        raise

# xAI Artifact Tags: #vial2 #tests #webhooks #neon_mcp
