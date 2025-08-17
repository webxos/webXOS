import uuid
import json
from config.config import DatabaseConfig
from postgrest import AsyncPostgrestClient
import logging

logger = logging.getLogger(__name__)

class WalletTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")
        self.project_id = "twilight-art-21036984"

    async def execute(self, args: dict) -> dict:
        method = args.get("method")
        user_id = args.get("user_id")
        project_id = args.get("project_id", self.project_id)
        if project_id != self.project_id:
            raise ValueError("Invalid Neon project ID")
        if method == "export":
            return await self.export_wallet(user_id, project_id)
        elif method == "import":
            return await self.import_wallet(user_id, args.get("markdown"), project_id)
        elif method == "transaction":
            return await self.process_transaction(user_id, args.get("args"), project_id)
        else:
            raise ValueError("Unknown wallet method")

    async def export_wallet(self, user_id: str, project_id: str) -> dict:
        try:
            wallet = await self.db.query(
                "SELECT wallet_id, address, balance FROM wallets WHERE user_id = $1 AND project_id = $2",
                [user_id, project_id]
            )
            if not wallet:
                raise ValueError("Wallet not found")
            wallet_data = {"wallet_id": wallet[0][0], "address": wallet[0][1], "balance": wallet[0][2]}
            transactions = await self.db.query(
                "SELECT transaction_id, transaction_type, amount, timestamp, metadata FROM wallet_transactions WHERE user_id = $1 AND project_id = $2",
                [user_id, project_id]
            )
            markdown = f"# Wallet Export\n\n**User ID**: {user_id}\n**Wallet**: {json.dumps(wallet_data)}\n**Transactions**: {json.dumps(transactions)}"
            logger.info(f"Wallet exported for user {user_id}")
            return {"status": "success", "markdown": markdown}
        except Exception as e:
            logger.error(f"Wallet export failed: {str(e)}")
            raise ValueError(f"Wallet export failed: {str(e)}")

    async def import_wallet(self, user_id: str, markdown: str, project_id: str) -> dict:
        try:
            lines = markdown.split("\n")
            wallet_data = json.loads(lines[2].split("**Wallet**: ")[1])
            transactions = json.loads(lines[3].split("**Transactions**: ")[1])
            await self.db.query(
                "INSERT INTO wallets (wallet_id, user_id, address, balance, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (user_id) DO UPDATE SET balance = EXCLUDED.balance",
                [wallet_data["wallet_id"], user_id, wallet_data["address"], wallet_data["balance"], str(uuid.uuid4()), project_id]
            )
            for tx in transactions:
                await self.db.query(
                    "INSERT INTO wallet_transactions (transaction_id, user_id, transaction_type, amount, timestamp, metadata, project_id) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    [tx[0], user_id, tx[1], tx[2], tx[3], tx[4], project_id]
                )
            self.data_api.auth(args.get("access_token"))
            await self.data_api.from_("wallets").insert({"wallet_id": wallet_data["wallet_id"], "user_id": user_id, "address": wallet_data["address"], "balance": wallet_data["balance"], "project_id": project_id}).eq("user_id", user_id).execute()
            logger.info(f"Wallet imported for user {user_id}")
            return {"status": "success", "vials": [{"id": "vial1", "status": "stopped"}]}
        except Exception as e:
            logger.error(f"Wallet import failed: {str(e)}")
            raise ValueError(f"Wallet import failed: {str(e)}")

    async def process_transaction(self, user_id: str, args: list, project_id: str) -> dict:
        try:
            transaction_type, amount = args[:2]
            amount = float(amount)
            transaction_id = str(uuid.uuid4())
            await self.db.query(
                "INSERT INTO wallet_transactions (transaction_id, user_id, transaction_type, amount, metadata, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [transaction_id, user_id, transaction_type, amount, json.dumps({"description": "WEBXOS transaction"}), project_id]
            )
            await self.db.query(
                "UPDATE wallets SET balance = balance + $1 WHERE user_id = $2 AND project_id = $3",
                [amount if transaction_type == "deposit" else -amount, user_id, project_id]
            )
            self.data_api.auth(args.get("access_token"))
            await self.data_api.from_("wallet_transactions").insert({"user_id": user_id, "transaction_type": transaction_type, "amount": amount, "project_id": project_id}).eq("user_id", user_id).execute()
            logger.info(f"Transaction {transaction_id} processed for user {user_id}")
            return {"status": "success", "transaction_id": transaction_id}
        except Exception as e:
            logger.error(f"Transaction processing failed: {str(e)}")
            raise ValueError(f"Transaction processing failed: {str(e)}")
