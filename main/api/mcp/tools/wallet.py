from config.config import DatabaseConfig
from lib.errors import ValidationError
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any, List
import hashlib
import re
import uuid

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

class WalletVoidInput(BaseModel):
    user_id: str
    vial_id: str

class WalletVoidOutput(BaseModel):
    vial_id: str
    status: str

class WalletTroubleshootInput(BaseModel):
    user_id: str
    vial_id: str

class WalletTroubleshootOutput(BaseModel):
    vial_id: str
    status: str
    diagnostics: Dict[str, Any]

class WalletQuantumLinkInput(BaseModel):
    user_id: str

class WalletQuantumLinkOutput(BaseModel):
    link_id: str

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
            elif method == "voidVial":
                void_input = WalletVoidInput(**input)
                return await self.void_vial(void_input)
            elif method == "troubleshootVial":
                troubleshoot_input = WalletTroubleshootInput(**input)
                return await self.troubleshoot_vial(troubleshoot_input)
            elif method == "quantumLink":
                quantum_input = WalletQuantumLinkInput(**input)
                return await self.quantum_link(quantum_input)
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
            
            balances = []
            for line in input.markdown.splitlines():
                if match := re.match(r".*balance\s*=\s*(\d+\.\d+)", line):
                    balances.append(float(match.group(1)))
            
            total_balance = sum(balances)
            current_balance = float(user.rows[0]["balance"])
            new_balance = current_balance + total_balance
            
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
            
            data = f"{input.user_id}{input.vial_id}{input.nonce}"
            hash_value = hashlib.sha256(data.encode()).hexdigest()
            difficulty = 2
            reward = 0.0
            
            if hash_value.startswith("0" * difficulty):
                reward = 1.0
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

    async def void_vial(self, input: WalletVoidInput) -> WalletVoidOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            # Simulate voiding by resetting balance for simplicity
            await self.db.query(
                "UPDATE users SET balance = 0 WHERE user_id = $1",
                [input.user_id]
            )
            
            logger.info(f"Voided vial {input.vial_id} for {input.user_id}")
            return WalletVoidOutput(vial_id=input.vial_id, status="voided")
        except Exception as e:
            logger.error(f"Void vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def troubleshoot_vial(self, input: WalletTroubleshootInput) -> WalletTroubleshootOutput:
        try:
            user = await self.db.query("SELECT user_id, balance, wallet_address FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            diagnostics = {
                "balance": float(user.rows[0]["balance"]),
                "wallet_address": user.rows[0]["wallet_address"],
                "active": user.rows[0]["balance"] > 0
            }
            
            logger.info(f"Troubleshooted vial {input.vial_id} for {input.user_id}")
            return WalletTroubleshootOutput(
                vial_id=input.vial_id,
                status="operational" if diagnostics["active"] else "inactive",
                diagnostics=diagnostics
            )
        except Exception as e:
            logger.error(f"Troubleshoot vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def quantum_link(self, input: WalletQuantumLinkInput) -> WalletQuantumLinkOutput:
        try:
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            link_id = str(uuid.uuid4())
            # Simulate quantum link by logging a unique connection ID
            logger.info(f"Established quantum link for {input.user_id}: {link_id}")
            return WalletQuantumLinkOutput(link_id=link_id)
        except Exception as e:
            logger.error(f"Quantum link error: {str(e)}")
            raise HTTPException(400, str(e))
