from pymongo import MongoClient
from datetime import datetime
from main.api.mcp.blockchain import Blockchain
from main.api.utils.logging import logger
import uuid

class TransactionManager:
    def __init__(self, mongo_url="mongodb://localhost:27017"):
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client["neon_mcp"]
        self.transactions_collection = self.db["transactions"]
        self.blockchain = Blockchain()

    def record_transaction(self, user_id: str, vial_id: str, amount: float, transaction_type: str):
        """Record a $WEBXOS transaction."""
        try:
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "vial_id": vial_id,
                "amount": amount,
                "type": transaction_type,
                "timestamp": datetime.now(),
                "block_hash": None
            }
            block_hash = self.blockchain.add_block({
                "type": transaction_type,
                "user_id": user_id,
                "vial_id": vial_id,
                "amount": amount
            })
            transaction["block_hash"] = block_hash
            self.transactions_collection.insert_one(transaction)
            logger.info(f"Transaction recorded: {transaction_type} for {vial_id}, amount: {amount}, block: {block_hash}")
            return transaction
        except Exception as e:
            logger.error(f"Transaction recording failed: {str(e)}")
            raise

    def get_transaction_history(self, user_id: str, vial_id: str = None):
        """Retrieve transaction history for a user or vial."""
        try:
            query = {"user_id": user_id}
            if vial_id:
                query["vial_id"] = vial_id
            transactions = list(self.transactions_collection.find(query).sort("timestamp", -1))
            logger.info(f"Retrieved {len(transactions)} transactions for user {user_id}")
            return transactions
        except Exception as e:
            logger.error(f"Transaction history retrieval failed: {str(e)}")
            raise
