from pymongo import MongoClient
from datetime import datetime
from main.api.mcp.blockchain import Blockchain
from main.api.utils.logging import logger
import uuid

class WalletManager:
    def __init__(self, mongo_url="mongodb://localhost:27017"):
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client["neon_mcp"]
        self.wallets_collection = self.db["wallets"]
        self.blockchain = Blockchain()

    def create_wallet(self, user_id: str):
        """Create a new $WEBXOS wallet for a user."""
        try:
            wallet = {
                "user_id": user_id,
                "address": str(uuid.uuid4()),
                "balance": 38940.0000,
                "created_at": datetime.now(),
                "transactions": []
            }
            self.wallets_collection.insert_one(wallet)
            block_hash = self.blockchain.add_block({
                "type": "wallet_create",
                "user_id": user_id,
                "address": wallet["address"]
            })
            logger.info(f"Wallet created for user {user_id}, block: {block_hash}")
            return wallet
        except Exception as e:
            logger.error(f"Wallet creation failed for user {user_id}: {str(e)}")
            raise

    def get_wallet(self, user_id: str):
        """Retrieve wallet details for a user."""
        try:
            wallet = self.wallets_collection.find_one({"user_id": user_id})
            if not wallet:
                return self.create_wallet(user_id)
            return wallet
        except Exception as e:
            logger.error(f"Wallet retrieval failed for user {user_id}: {str(e)}")
            raise

    def update_balance(self, user_id: str, amount: float, transaction_type: str):
        """Update wallet balance with a transaction."""
        try:
            wallet = self.get_wallet(user_id)
            new_balance = wallet["balance"] + amount
            if new_balance < 0:
                raise ValueError("Insufficient $WEBXOS balance")
            self.wallets_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {"balance": new_balance},
                    "$push": {
                        "transactions": {
                            "type": transaction_type,
                            "amount": amount,
                            "timestamp": datetime.now()
                        }
                    }
                }
            )
            block_hash = self.blockchain.add_block({
                "type": transaction_type,
                "user_id": user_id,
                "amount": amount
            })
            logger.info(f"Balance updated for user {user_id}: {amount}, block: {block_hash}")
            return new_balance
        except Exception as e:
            logger.error(f"Balance update failed for user {user_id}: {str(e)}")
            raise

    def distribute_to_vials(self, user_id: str, amount: float):
        """Distribute $WEBXOS to four vials."""
        try:
            per_vial = amount / 4
            wallet = self.get_wallet(user_id)
            self.wallets_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {"balance": wallet["balance"] - amount},
                    "$push": {
                        "transactions": {
                            "type": "vial_distribution",
                            "amount": -amount,
                            "vial_amount": per_vial,
                            "timestamp": datetime.now()
                        }
                    }
                }
            )
            block_hash = self.blockchain.add_block({
                "type": "vial_distribution",
                "user_id": user_id,
                "amount": amount,
                "per_vial": per_vial
            })
            logger.info(f"Distributed {amount} to vials for user {user_id}, block: {block_hash}")
            return per_vial
        except Exception as e:
            logger.error(f"Vial distribution failed for user {user_id}: {str(e)}")
            raise
