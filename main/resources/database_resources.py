from pymongo import MongoClient
from ..config.settings import settings

class DatabaseResource:
    def __init__(self):
        self.client = MongoClient(settings.database.url)
        self.db = self.client[settings.database.db_name]
    
    def get_wallet(self, user_id: str):
        return self.db.wallets.find_one({"user_id": user_id})
    
    def update_wallet(self, user_id: str, wallet_data: dict):
        self.db.wallets.update_one(
            {"user_id": user_id},
            {"$set": wallet_data},
            upsert=True
        )
    
    def get_quantum_state(self, user_id: str):
        return self.db.quantum_states.find_one({"user_id": user_id})
    
    def update_quantum_state(self, user_id: str, quantum_data: dict):
        self.db.quantum_states.update_one(
            {"user_id": user_id},
            {"$set": quantum_data},
            upsert=True
        )
    
    def close(self):
        self.client.close()
