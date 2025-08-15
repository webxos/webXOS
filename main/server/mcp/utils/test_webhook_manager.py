# main/server/mcp/utils/test_webhook_manager.py
import pytest
from fastapi.testclient import TestClient
from ..utils.webhook_manager import WebhookManager, router
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
async def webhook_manager():
    manager = WebhookManager()
    yield manager
    await manager.redis_client.flushdb()
    await manager.close()

@pytest.mark.asyncio
async def test_register_webhook(webhook_manager):
    webhook_id = await webhook_manager.register_webhook(
        user_id="test_user",
        endpoint="https://example.com/webhook",
        event_types=["note_created", "agent_updated"]
    )
    assert webhook_id in webhook_manager.webhooks
    assert webhook_manager.webhooks[webhook_id]["user_id"] == "test_user"
    assert webhook_manager.webhooks[webhook_id]["endpoint"] == "https://example.com/webhook"

@pytest.mark.asyncio
async def test_register_webhook_invalid(webhook_manager):
    with pytest.raises(MCPError) as exc_info:
        await webhook_manager.register_webhook("test_user", "", ["note_created"])
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Endpoint and event types are required"

@pytest.mark.asyncio
async def test_publish_event(webhook_manager, mocker):
    mocker.patch("redis.asyncio.Redis.publish")
    webhook_id = await webhook_manager.register_webhook(
        user_id="test_user",
        endpoint="https://example.com/webhook",
        event_types=["note_created"]
    )
    await webhook_manager.publish_event("note_created", {"note_id": "123"})
    webhook_manager.redis_client.publish.assert_called_once()

@pytest.mark.asyncio
async def test_deregister_webhook(webhook_manager):
    webhook_id = await webhook_manager.register_webhook(
        user_id="test_user",
        endpoint="https://example.com/webhook",
        event_types=["note_created"]
    )
    await webhook_manager.deregister_webhook(webhook_id, "test_user")
    assert webhook_id not in webhook_manager.webhooks
    assert await webhook_manager.redis_client.get(f"webhook:{webhook_id}") is None

@pytest.mark.asyncio
async def test_deregister_webhook_invalid(webhook_manager):
    with pytest.raises(MCPError) as exc_info:
        await webhook_manager.deregister_webhook("invalid_id", "test_user")
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Webhook not found or access denied"
