from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pymongo
import logging
import datetime
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalletRequest(BaseModel):
    user_id: str
    transaction: dict

class WebXOSWallet:
    def __init__(self):
        self.mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["mcp_db"]
        self.sqlite_conn = sqlite3.connect("vial/database.sqlite")
        self.cursor = self.sqlite_conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                user_id TEXT PRIMARY KEY,
                balance REAL,
                transactions TEXT
            )
        """)
        self.sqlite_conn.commit()

    async def update_wallet(self, user_id: str, transaction: dict) -> dict:
        try:
            wallet = self.db.collection("wallet").find_one({"user_id": user_id}) or {"webxos": 0.0, "transactions": []}
            wallet["transactions"].append({
                **transaction,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            
            # Update MongoDB
            self.db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            # Update SQLite
            self.cursor.execute("""
                INSERT OR REPLACE INTO wallet (user_id, balance, transactions)
                VALUES (?, ?, ?)
            """, (user_id, wallet["webxos"], json.dumps(wallet["transactions"])))
            self.sqlite_conn.commit()
            
            return wallet
        except Exception as e:
            logger.error(f"Wallet update error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Wallet update error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

wallet_manager = WebXOSWallet()

@app.post("/api/update_wallet")
async def update_wallet(request: WalletRequest):
    return await wallet_manager.update_wallet(request.user_id, request.transaction)
