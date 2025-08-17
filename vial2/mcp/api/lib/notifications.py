from fastapi import WebSocket
from lib.security import SecurityHandler
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class NotificationHandler:
    def __init__(self):
        self.active_connections = {}
        self.security = SecurityHandler(DatabaseConfig())

    async def handle_websocket(self, websocket: WebSocket, token: str):
        try:
            await websocket.accept()
            decoded = await self.security.verify_jwt(token)
            if "error" in decoded:
                await websocket.send_json({"error": decoded["error"]})
                await websocket.close()
                logger.error(f"WebSocket auth failed: {decoded['error']} [notifications.py:20] [ID:ws_auth_error]")
                return
            user_id = decoded.get("sub")
            self.active_connections[user_id] = websocket
            logger.info(f"WebSocket connected for user {user_id} [notifications.py:25] [ID:ws_connect_success]")
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.broadcast(user_id, data)
            except Exception as e:
                logger.error(f"WebSocket disconnected: {str(e)} [notifications.py:30] [ID:ws_disconnect_error]")
            finally:
                self.active_connections.pop(user_id, None)
                await websocket.close()
                logger.info(f"WebSocket closed for user {user_id} [notifications.py:35] [ID:ws_close_success]")
        except Exception as e:
            logger.error(f"WebSocket setup failed: {str(e)} [notifications.py:40] [ID:ws_setup_error]")
            await websocket.close()

    async def broadcast(self, user_id: str, message: str):
        try:
            data = json.loads(message)
            for conn_user_id, conn in self.active_connections.items():
                if conn_user_id == user_id:
                    await conn.send_json(data)
            logger.info(f"Broadcast sent to user {user_id} [notifications.py:45] [ID:broadcast_success]")
        except Exception as e:
            logger.error(f"Broadcast failed: {str(e)} [notifications.py:50] [ID:broadcast_error]")
            raise
