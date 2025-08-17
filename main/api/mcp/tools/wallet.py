from config.config import DatabaseConfig
from lib.errors import ValidationError
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any

logger = logging.getLogger("mcp.wallet")
logger.setLevel(logging.INFO)

class WalletBalanceInput(BaseModel):
    user_id: str
    vial_id: str

class WalletBalanceOutput(BaseModel):
    vial_id: str
    balance: float

class WalletTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "getVialBalance")
            if method == "getVialBalance":
                wallet_input = WalletBalanceInput(**input)
                return await self.get_vial_balance(wallet_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Wallet error: {str(e)}")
            raise HTTPException(400, str(e))

    async def get_vial_balance(self, input: WalletBalanceInput) -> WalletBalanceOutput:
        try:
            # Validate user exists
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")

            # Simulate Claude-generated wallet logic (based on vial_wallet_export)
            # Assume vial_id corresponds to VialAgent1-4; fetch balance from users table
            vial_balance = await self.db.query(
                "SELECT balance FROM users WHERE user_id = $1",
                [input.user_id]
            )
            if not vial_balance.rows:
                raise ValidationError(f"No balance data for user: {input.user_id}")

            balance = float(vial_balance.rows[0]["balance"])
            logger.info(f"Retrieved vial balance for {input.user_id}, vial {input.vial_id}: {balance}")
            return WalletBalanceOutput(vial_id=input.vial_id, balance=balance)
        except Exception as e:
            logger.error(f"Get vial balance error: {str(e)}")
            raise HTTPException(400, str(e))
