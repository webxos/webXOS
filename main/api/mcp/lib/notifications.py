import json
import logging
from typing import Dict, Any
from fastapi import WebSocket
from config.config import DatabaseConfig

logger = logging.getLogger("mcp.notifications")
logger.setLevel(logging.INFO)

class NotificationHandler:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        await self.notify(client_id, {
            "method": "connection",
            "params": {"status": "connected", "client_id": client_id}
        })
        logger.info(f"WebSocket connected for client {client_id}")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].close()
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected for client {client_id}")

    async def notify(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending notification to {client_id}: {str(e)}")
                await self.disconnect(client_id)

    async def notify_auth(self, user_id: str, access_token: str):
        await self.notify(user_id, {
            "method": "auth.success",
            "params": {"user_id": user_id, "access_token": access_token}
        })
        logger.info(f"Notified auth success for {user_id}")

    async def notify_git_push(self, user_id: str, vial_id: str, commit_hash: str, balance: float):
        await self.notify(user_id, {
            "method": "vial_management.gitPush",
            "params": {"vial_id": vial_id, "commit_hash": commit_hash, "balance": balance}
        })
        logger.info(f"Notified Git push for {user_id}, vial {vial_id}")

    async def notify_cash_out(self, user_id: str, transaction_id: str, amount: float, new_balance: float):
        await self.notify(user_id, {
            "method": "wallet.cashOut",
            "params": {"transaction_id": transaction_id, "amount": amount, "new_balance": new_balance}
        })
        logger.info(f"Notified cash-out for {user_id}, transaction {transaction_id}")

    async def notify_wallet_update(self, user_id: str, vial_id: str, balance: float):
        await self.notify(user_id, {
            "method": "wallet.update",
            "params": {"vial_id": vial_id, "balance": balance}
        })
        logger.info(f"Notified wallet update for {user_id}, vial {vial_id}")
