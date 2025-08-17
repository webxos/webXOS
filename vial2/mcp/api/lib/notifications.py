from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class NotificationHandler:
    def __init__(self):
        self.connections = []
        self.project_id = "twilight-art-21036984"

    async def handle_websocket(self, websocket: WebSocket, token: str):
        await websocket.accept()
        self.connections.append(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                project_id = data.get("project_id", self.project_id)
                if project_id != self.project_id:
                    await websocket.send_json({"error": "Invalid Neon project ID"})
                    continue
                message = {"type": "notification", "data": data, "timestamp": datetime.now().isoformat()}
                await self.broadcast(message)
                logger.info(f"Notification sent: {data}")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            self.connections.remove(websocket)
            await websocket.close()

    async def broadcast(self, message: dict):
        for connection in self.connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast failed: {str(e)}")
                self.connections.remove(connection)
