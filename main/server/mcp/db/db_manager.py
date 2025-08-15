# main/server/mcp/db/db_manager.py
from pymongo import MongoClient
from typing import Dict, Any
from ..utils.mcp_error_handler import MCPError

class DBManager:
    def __init__(self, uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(uri)
        self.db = self.client["vial_mcp"]

    async def save_user(self, username: str, password: str, vials: Dict) -> bool:
        try:
            users = self.db.users
            users.insert_one({"username": username, "password": password, "vials": vials})
            return True
        except Exception as e:
            raise MCPError(code=-32603, message=f"DB save failed: {str(e)}")
