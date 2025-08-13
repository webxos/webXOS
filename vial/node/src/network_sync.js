import logging
import sqlite3
from pymongo import MongoClient
from vial.webxos_wallet import WebXOSWallet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkSync:
    def __init__(self):
        self.db = MongoClient('mongodb://mongo:27017')['mcp_db']
        self.wallet = WebXOSWallet()

    def sync_wallet(self, user_id: str) -> bool:
        try:
            local_balance = self.wallet.get_balance(user_id)
            remote_wallet = self.db.wallet.find_one({"userId": user_id}) or {"webxos": 0.0}
            if local_balance != remote_wallet["webxos"]:
                self.db.wallet.update_one(
                    {"userId": user_id},
                    {"$set": {"webxos": local_balance}},
                    upsert=True
                )
                self.wallet.update_balance(user_id, remote_wallet["webxos"])
            logger.info(f"Synced wallet for {user_id}: {local_balance} $WEBXOS")
            return True
        except Exception as e:
            logger.error(f"Wallet sync error for {user_id}: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Wallet sync error: {str(e)}\n")
            return False

    def sync_vials(self, vials: dict) -> bool:
        try:
            for vial_id, vial_data in vials.items():
                self.db.vials.update_one(
                    {"id": vial_id},
                    {"$set": vial_data},
                    upsert=True
                )
            logger.info(f"Synced {len(vials)} vials")
            return True
        except Exception as e:
            logger.error(f"Vial sync error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Vial sync error: {str(e)}\n")
            return False
