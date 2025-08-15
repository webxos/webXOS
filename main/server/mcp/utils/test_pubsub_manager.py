# main/server/mcp/utils/test_pubsub_manager.py
import pytest
from fastapi.testclient import TestClient
from ..utils.pubsub_manager import PubSubManager, router
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
async def pubsub_manager():
    manager = PubSubManager()
    yield manager
    await manager.redis_client.flushdb()
    await manager.close()

@pytest.mark.asyncio
async def test_subscribe(pubsub_manager):
    subscription_id = await pubsub_manager.subscribe("test_user", "notes_channel", ["note_created"])
    assert subscription_id in pubsub_manager.subscriptions
    assert pubsub_manager.subscriptions[subscription_id]["user_id"] == "test_user"
    assert pubsub_manager.subscriptions[subscription_id]["channel"] == "notes_channel"

@pytest.mark.asyncio
async def test_subscribe_invalid(pubsub_manager):
    with pytest.raises(MCPError) as exc_info:
        await pubsub_manager.subscribe("", "notes_channel", ["note_created"])
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "User ID, channel, and event types are required"

@pytest.mark.asyncio
async def test_publish(pubsub_manager, mocker):
    mocker.patch("redis.asyncio.Redis.publish")
    subscription_id = await pubsub_manager.subscribe("test_user", "notes_channel", ["note_created"])
    await pubsub_manager.publish("notes_channel", "note_created", {"note_id": "123"})
    pubsub_manager.redis_client.publish.assert_called_once()

@pytest.mark.asyncio
async def test_unsubscribe(pubsub_manager):
    subscription_id = await pubsub_manager.subscribe("test_user", "notes_channel", ["note_created"])
    await pubsub_manager.unsubscribe(subscription_id, "test_user")
    assert subscription_id not in pubsub_manager.subscriptions
    assert await pubsub_manager.redis_client.get(f"subscription:{subscription_id}") is None

@pytest.mark.asyncio
async def test_listen(pubsub_manager, mocker):
    subscription_id = await pubsub_manager.subscribe("test_user", "notes_channel", ["note_created"])
    mock_pubsub = mocker.patch("redis.asyncio.Redis.pubsub")
    mock_pubsub.return_value.listen.return_value = [
        {"type": "message", "data": json.dumps({"event_type": "note_created", "data": {"note_id": "123"}}).encode()}
    ]
    messages = []
    async for message in pubsub_manager.listen(subscription_id, "test_user"):
        messages.append(message)
        break
    assert len(messages) == 1
    assert messages[0]["event_type"] == "note_created"
