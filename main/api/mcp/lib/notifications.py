from pydantic import BaseModel
from typing import Dict, Any
import logging
from fastapi import WebSocket
from lib.mcp_protocol import MCPNotification

logger = logging.getLogger("mcp.notifications")
logger.setLevel(logging.INFO)

class NotificationHandler:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].close()
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: {client_id}")

    async def send_notification(self, client_id: str, notification: MCPNotification):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(notification.dict(exclude_none=True))
                logger.info(f"Notification sent to {client_id}: {notification.method}")
            except Exception as e:
                logger.error(f"Failed to send notification to {client_id}: {str(e)}")
                await self.disconnect(client_id)
