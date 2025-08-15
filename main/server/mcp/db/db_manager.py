# main/server/mcp/db/db_manager.py
from pymongo import MongoClient
from typing import Any, Dict, List, Optional
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
import os

class DBManager:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.metrics = PerformanceMetrics()

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        with self.metrics.track_span("db_insert_one", {"collection": collection}):
            try:
                result = self.db[collection].insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                handle_generic_error(e, context=f"db_insert_one_{collection}")
                raise

    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self.metrics.track_span("db_find_one", {"collection": collection}):
            try:
                return self.db[collection].find_one(query)
            except Exception as e:
                handle_generic_error(e, context=f"db_find_one_{collection}")
                raise

    def find_many(self, collection: str, query: Dict[str, Any], limit: int = 0, skip: int = 0) -> List[Dict[str, Any]]:
        with self.metrics.track_span("db_find_many", {"collection": collection, "limit": limit, "skip": skip}):
            try:
                return list(self.db[collection].find(query).skip(skip).limit(limit))
            except Exception as e:
                handle_generic_error(e, context=f"db_find_many_{collection}")
                raise

    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        with self.metrics.track_span("db_update_one", {"collection": collection}):
            try:
                result = self.db[collection].update_one(query, {"$set": update})
                return result.modified_count
            except Exception as e:
                handle_generic_error(e, context=f"db_update_one_{collection}")
                raise

    def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        with self.metrics.track_span("db_delete_one", {"collection": collection}):
            try:
                result = self.db[collection].delete_one(query)
                return result.deleted_count
            except Exception as e:
                handle_generic_error(e, context=f"db_delete_one_{collection}")
                raise
