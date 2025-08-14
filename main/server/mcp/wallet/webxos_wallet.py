# main/server/mcp/wallet/webxos_wallet.py
from web3 import Web3
from pymongo import MongoClient
from fastapi import HTTPException
from ..utils.error_handler import handle_wallet_error
import os
from datetime import datetime

class WebXOSWallet:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER", "http://localhost:8545")))
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.wallets_collection = self.db["wallets"]

    async def create_wallet(self, user_id: str):
        try:
            account = self.web3.eth.account.create()
            wallet_data = {
                "user_id": user_id,
                "address": account.address,
                "balance": 0.0,
                "hash": self.web3.keccak(text=account.address).hex(),
                "webxos": 0.0,
                "transactions": [{"type": "created", "timestamp": datetime.utcnow()}]
            }
            self.wallets_collection.insert_one(wallet_data)
            return wallet_data
        except Exception as e:
            handle_wallet_error(e)
            raise HTTPException(status_code=500, detail="Failed to create wallet")

    async def import_wallet(self, user_id: str, wallet_data: str):
        try:
            lines = wallet_data.split("\n")
            new_wallet = {"user_id": user_id, "balance": 0.0, "webxos": 0.0, "transactions": []}
            for line in lines:
                if line.startswith("- Address: "):
                    new_wallet["address"] = line.split(": ")[1]
                elif line.startswith("- Balance: "):
                    new_wallet["webxos"] = float(line.split(": ")[1].split(" ")[0])
                elif line.startswith("- Hash: "):
                    new_wallet["hash"] = line.split(": ")[1]
                elif line.startswith("- Transactions: "):
                    new_wallet["transactions"] = eval(line.split(": ")[1]) if line.split(": ")[1] else []
            if not new_wallet.get("address") or not new_wallet.get("hash"):
                raise ValueError("Invalid wallet data")
            new_wallet["transactions"].append({"type": "imported", "timestamp": datetime.utcnow()})
            self.wallets_collection.update_one({"user_id": user_id}, {"$set": new_wallet}, upsert=True)
            return new_wallet
        except Exception as e:
            handle_wallet_error(e)
            raise HTTPException(status_code=400, detail=f"Failed to import wallet: {str(e)}")

    async def verify_wallet(self, user_id: str, address: str):
        try:
            wallet = self.wallets_collection.find_one({"user_id": user_id, "address": address})
            if not wallet:
                raise ValueError("Wallet not found")
            return wallet
        except Exception as e:
            handle_wallet_error(e)
            raise HTTPException(status_code=404, detail="Wallet verification failed")
