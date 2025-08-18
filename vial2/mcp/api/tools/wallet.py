from config.config import DatabaseConfig
from lib.security import SecurityHandler
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class WalletTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.project_id = db.project_id

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            user_id = data.get("user_id")
            project_id = data.get("project_id", self.project_id)
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [wallet.py:20] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            if method == "transaction":
                return await self.process_transaction(user_id, data.get("transaction_type"), data.get("amount"), project_id)
            elif method == "validate_md_wallet":
                return await self.validate_md_wallet(user_id, data.get("wallet_data"), project_id)
            else:
                error_message = f"Invalid wallet method: {method} [wallet.py:25] [ID:wallet_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Wallet operation failed: {str(e)} [wallet.py:30] [ID:wallet_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "wallet", error_message)
            return {"error": error_message}

    async def process_transaction(self, user_id: str, transaction_type: str, amount: float, project_id: str) -> dict:
        try:
            if transaction_type not in ["deposit", "withdraw"]:
                error_message = f"Invalid transaction type: {transaction_type} [wallet.py:35] [ID:transaction_type_error]"
                logger.error(error_message)
                return {"error": error_message}
            await self.db.query(
                "INSERT INTO wallet_transactions (transaction_id, user_id, type, amount, project_id) VALUES ($1, $2, $3, $4, $5)",
                [str(uuid.uuid4()), user_id, transaction_type, amount, project_id]
            )
            logger.info(f"Transaction processed: {transaction_type} {amount} for user: {user_id} [wallet.py:40] [ID:transaction_success]")
            return {"status": "success", "transaction_type": transaction_type, "amount": amount}
        except Exception as e:
            error_message = f"Transaction failed: {str(e)} [wallet.py:45] [ID:transaction_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def validate_md_wallet(self, user_id: str, wallet_data: dict, project_id: str) -> dict:
        try:
            if not wallet_data.get("address") or not wallet_data.get("signature"):
                error_message = "Invalid .md wallet data [wallet.py:50] [ID:md_wallet_validation_error]"
                logger.error(error_message)
                return {"error": error_message}
            # Simulate .md wallet validation (replace with actual $WEBXOS wallet logic)
            is_valid = wallet_data["address"].startswith("0x") and len(wallet_data["signature"]) > 10
            if not is_valid:
                error_message = f"Invalid .md wallet: {wallet_data['address']} [wallet.py:55] [ID:md_wallet_invalid_error]"
                logger.error(error_message)
                return {"error": error_message}
            await self.db.query(
                "INSERT INTO wallet_transactions (transaction_id, user_id, type, data, project_id) VALUES ($1, $2, $3, $4, $5)",
                [str(uuid.uuid4()), user_id, "md_wallet_validation", json.dumps(wallet_data), project_id]
            )
            logger.info(f".md wallet validated for user: {user_id} [wallet.py:60] [ID:md_wallet_validation_success]")
            return {"status": "success", "wallet_address": wallet_data["address"]}
        except Exception as e:
            error_message = f".md wallet validation failed: {str(e)} [wallet.py:65] [ID:md_wallet_validation_error]"
            logger.error(error_message)
            return {"error": error_message}
