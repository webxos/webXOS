import pytest
from unittest.mock import AsyncMock
from lib.notifications import NotificationHandler
import websockets
import json

@pytest.fixture
async def notification_handler():
    return NotificationHandler()

@pytest.mark.asyncio
async def test_connect_disconnect(notification_handler):
    websocket = AsyncMock(spec=websockets.WebSocketServerProtocol)
    client_id = "user_12345"
    
    await notification_handler.connect(websocket, client_id)
    assert client_id in notification_handler.clients
    assert websocket in notification_handler.clients[client_id]
    
    await notification_handler.disconnect(client_id)
    assert client_id not in notification_handler.clients

@pytest.mark.asyncio
async def test_send_notification_success(notification_handler):
    websocket = AsyncMock(spec=websockets.WebSocketServerProtocol)
    client_id = "user_12345"
    message = {
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {"imported_vials": ["vial1"], "total_balance": 150.0}
    }
    
    await notification_handler.connect(websocket, client_id)
    await notification_handler.send_notification(client_id, message)
    
    websocket.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_send_notification_offline_import_sync(notification_handler):
    websocket = AsyncMock(spec=websockets.WebSocketServerProtocol)
    client_id = "user_12345"
    message = {
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {"imported_vials": ["vial1"], "total_balance": 150.0}
    }
    
    await notification_handler.connect(websocket, client_id)
    await notification_handler.send_notification(client_id, message)
    
    websocket.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_send_notification_offline_mining_sync(notification_handler):
    websocket = AsyncMock(spec=websockets.WebSocketServerProtocol)
    client_id = "user_12345"
    message = {
        "jsonrpc": "2.0",
        "method": "wallet.mineVial",
        "params": {"hash": "00abc123", "reward": 1.0, "balance": 101.0}
    }
    
    await notification_handler.connect(websocket, client_id)
    await notification_handler.send_notification(client_id, message)
    
    websocket.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_notification_error_handling(notification_handler):
    websocket = AsyncMock(spec=websockets.WebSocketServerProtocol)
    websocket.send.side_effect = Exception("Send error")
    client_id = "user_12345"
    message = {
        "jsonrpc": "2.0",
        "method": "wallet.importWallet",
        "params": {"imported_vials": ["vial1"], "total_balance": 150.0}
    }
    
    await notification_handler.connect(websocket, client_id)
    await notification_handler.send_notification(client_id, message)
    
    assert client_id not in notification_handler.clients
