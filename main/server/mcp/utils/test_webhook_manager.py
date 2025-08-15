# main/server/mcp/utils/test_webhook_manager.py
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request
from ..utils.webhook_manager import WebhookManager, router
from ..utils.mcp_error_handler import MCPError
import hmac
import hashlib
import json

app = FastAPI()
app.include_router(router)

@pytest.fixture
async def webhook_manager():
    manager = WebhookManager()
    yield manager
    await manager.cache.redis_client.flushdb()
    await manager.close()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_register_webhook(webhook_manager):
    webhook_id = await webhook_manager.register_webhook("test_user", "https://example.com/webhook", ["note_created"])
    assert webhook_id in webhook_manager.webhooks
    assert webhook_manager.webhooks[webhook_id]["user_id"] == "test_user"
    cached = await webhook_manager.cache.get_cache(f"webhook:{webhook_id}")
    assert cached["url"] == "https://example.com/webhook"

@pytest.mark.asyncio
async def test_register_webhook_invalid(webhook_manager):
    with pytest.raises(MCPError) as exc_info:
        await webhook_manager.register_webhook("", "https://example.com/webhook", ["note_created"])
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "User ID, URL, and events are required"

@pytest.mark.asyncio
async def test_trigger_webhook(webhook_manager, mocker):
    mocker.patch("aiohttp.ClientSession.post", return_value=mocker.AsyncMock(status=200))
    webhook_id = await webhook_manager.register_webhook("test_user", "https://example.com/webhook", ["note_created"])
    await webhook_manager.trigger_webhook("note_created", {"note_id": "123"})
    assert webhook_manager.metrics.requests_total.labels(endpoint="trigger_webhook")._value.get() == 1

@pytest.mark.asyncio
async def test_verify_webhook(webhook_manager, mocker):
    payload = {"note_id": "123"}
    signature = hmac.new(
        webhook_manager.webhook_secret.encode(),
        json.dumps(payload, sort_keys=True).encode(),
        hashlib.sha256
    ).hexdigest()
    request = mocker.Mock(headers={"X-Webhook-Signature": signature})
    result = await webhook_manager.verify_webhook(request, payload)
    assert result is True

@pytest.mark.asyncio
async def test_delete_webhook(webhook_manager):
    webhook_id = await webhook_manager.register_webhook("test_user", "https://example.com/webhook", ["note_created"])
    await webhook_manager.delete_webhook(webhook_id, "test_user")
    assert webhook_id not in webhook_manager.webhooks
    assert await webhook_manager.cache.get_cache(f"webhook:{webhook_id}") is None
