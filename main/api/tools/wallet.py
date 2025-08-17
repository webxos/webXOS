from config.config import DatabaseConfig
from lib.errors import ValidationError
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any, List
import hashlib
import re

logger = logging.getLogger("mcp.wallet")
logger.setLevel(logging.INFO)

class WalletBalanceInput(BaseModel):
    user_id: str
    vial_id: str

class WalletBalanceOutput(BaseModel):
    vial_id: str
    balance: float

class WalletImportInput(BaseModel):
    user_id: str
    markdown: str

class WalletImportOutput(BaseModel):
    imported_vials: List[str]
    total_balance: float

class WalletExportOutput(BaseModel):
    markdown: str

class WalletMineInput(BaseModel):
    user_id: str
    vial_id: str
    nonce: int

class WalletMineOutput(BaseModel):
    hash: str
    reward: float

class WalletTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "getVialBalance")
            if method == "getVialBalance":
                wallet_input = WalletBalanceInput(**input)
                return await self.get_vial_balance(wallet_input)
            elif method == "importWallet":
                import_input = WalletImportInput(**input)
                return await self.import_wallet(import_input)
            elif method == "exportVials":
                export_input = WalletBalanceInput(**input)
                return await self.export_vials(export_input)
            elif method == "mineVial":
                mine_input = WalletMineInput(**input)
                return await self.mine_vial(mine_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Wallet error: {str(e)}")
            raise HTTPException(400, str(e))

    async def get_vial_balance(self, input: WalletBalanceInput) -> WalletBalanceOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            balance = float(user.rows[0]["balance"])
            logger.info(f"Retrieved vial balance for {input.user_id}, vial {input.vial_id}: {balance}")
            return WalletBalanceOutput(vial_id=input.vial_id, balance=balance)
        except Exception as e:
            logger.error(f"Get vial balance error: {str(e)}")
            raise HTTPException(400, str(e))

    async def import_wallet(self, input: WalletImportInput) -> WalletImportOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            # Parse markdown for vial balances (simplified from vial_wallet_export)
            balances = []
            for line in input.markdown.splitlines():
                if match := re.match(r".*balance\s*=\s*(\d+\.\d+)", line):
                    balances.append(float(match.group(1)))
            
            total_balance = sum(balances)
            current_balance = float(user.rows[0]["balance"])
            new_balance = current_balance + total_balance
            
            # Update user balance
            await self.db.query(
                "UPDATE users SET balance = $1 WHERE user_id = $2",
                [new_balance, input.user_id]
            )
            
            logger.info(f"Imported wallet for {input.user_id}, new balance: {new_balance}")
            return WalletImportOutput(
                imported_vials=[f"vial{i+1}" for i in range(len(balances))],
                total_balance=new_balance
            )
        except Exception as e:
            logger.error(f"Import wallet error: {str(e)}")
            raise HTTPException(400, str(e))

    async def export_vials(self, input: WalletBalanceInput) -> WalletExportOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            balance = float(user.rows[0]["balance"])
            markdown = f"""# Wallet Export for {input.user_id}
## Vial Balances
- {input.vial_id}: {balance}
"""
            logger.info(f"Exported vials for {input.user_id}")
            return WalletExportOutput(markdown=markdown)
        except Exception as e:
            logger.error(f"Export vials error: {str(e)}")
            raise HTTPException(400, str(e))

    async def mine_vial(self, input: WalletMineInput) -> WalletMineOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            # Simplified PoW: Find a hash with leading zeros
            data = f"{input.user_id}{input.vial_id}{input.nonce}"
            hash_value = hashlib.sha256(data.encode()).hexdigest()
            difficulty = 2  # Number of leading zeros
            reward = 0.0
            
            if hash_value.startswith("0" * difficulty):
                reward = 1.0  # Reward for valid PoW
                current_balance = float(user.rows[0]["balance"])
                await self.db.query(
                    "UPDATE users SET balance = $1 WHERE user_id = $2",
                    [current_balance + reward, input.user_id]
                )
                logger.info(f"Mining successful for {input.user_id}, vial {input.vial_id}, reward: {reward}")
            
            return WalletMineOutput(hash=hash_value, reward=reward)
        except Exception as e:
            logger.error(f"Mine vial error: {str(e)}")
            raise HTTPException(400, str(e))
