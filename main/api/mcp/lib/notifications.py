from typing import Dict, Any
import websockets
import json
import logging

logger = logging.getLogger("mcp.notifications")
logger.setLevel(logging.INFO)

class NotificationHandler:
    def __init__(self):
        self.clients: Dict[str, set] = {}

    async def connect(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        if client_id not in self.clients:
            self.clients[client_id] = set()
        self.clients[client_id].add(websocket)
        logger.info(f"Client connected: {client_id}")

    async def disconnect(self, client_id: str):
        if client_id in self.clients:
            self.clients[client_id].clear()
            del self.clients[client_id]
        logger.info(f"Client disconnected: {client_id}")

    async def send_notification(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.clients:
            for websocket in self.clients[client_id]:
                try:
                    await websocket.send(json.dumps(message))
                    logger.info(f"Sent notification to {client_id}: {message}")
                except Exception as e:
                    logger.error(f"Error sending notification to {client_id}: {str(e)}")
                    self.clients[client_id].remove(websocket)
                    if not self.clients[client_id]:
                        del self.clients[client_id]
