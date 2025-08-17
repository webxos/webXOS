import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocket
from lib.notifications import NotificationHandler
from lib.mcp_protocol import MCPNotification
from unittest.mock import AsyncMock, patch

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_notification_handler_connect_disconnect():
    handler = NotificationHandler()
    websocket = AsyncMock(spec=WebSocket)
    
    await handler.connect(websocket, "user_12345")
    assert "user_12345" in handler.active_connections
    await handler.disconnect("user_12345")
    assert "user_12345" not in handler.active_connections

@pytest.mark.asyncio
async def test_notification_handler_send_notification():
    handler = NotificationHandler()
    websocket = AsyncMock(spec=WebSocket)
    await handler.connect(websocket, "user_12345")
    
    notification = MCPNotification(method="claude.executionComplete", params={"output": "Test output"})
    await handler.send_notification("user_12345", notification)
    
    websocket.send_json.assert_called_once_with(notification.dict(exclude_none=True))

@pytest.mark.asyncio
async def test_notification_handler_send_to_disconnected():
    handler = NotificationHandler()
    notification = MCPNotification(method="claude.executionComplete", params={"output": "Test output"})
    await handler.send_notification("user_12345", notification)
    # No exception should be raised for non-existent client
