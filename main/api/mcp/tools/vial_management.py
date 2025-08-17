from config.config import DatabaseConfig
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any

logger = logging.getLogger("mcp.vial_management")
logger.setLevel(logging.INFO)

class UserDataOutput(BaseModel):
    user_id: str
    balance: float
    reputation: int
    wallet_address: str

class VialManagementTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "getUserData")
            if method == "getUserData":
                user_id = input.get("user_id")
                if not user_id:
                    raise HTTPException(400, "user_id is required")
                return await self.get_user_data(user_id)
            else:
                raise HTTPException(400, f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Vial management error: {str(e)}")
            raise HTTPException(400, str(e))

    async def get_user_data(self, user_id: str) -> UserDataOutput:
        try:
            user = await self.db.query(
                "SELECT user_id, balance, reputation, wallet_address FROM users WHERE user_id = $1",
                [user_id]
            )
            if not user.rows:
                raise HTTPException(404, f"User not found: {user_id}")
            user_data = user.rows[0]
            logger.info(f"Retrieved user data for {user_id}")
            return UserDataOutput(
                user_id=user_data["user_id"],
                balance=float(user_data["balance"]),
                reputation=int(user_data["reputation"]),
                wallet_address=user_data["wallet_address"]
            )
        except Exception as e:
            logger.error(f"Get user data error: {str(e)}")
            raise HTTPException(400, str(e))
