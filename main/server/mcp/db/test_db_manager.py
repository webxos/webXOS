# main/server/mcp/db/test_db_manager.py
import unittest
from unittest.mock import patch
from .db_manager import DBManager

class TestDBManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DBManager()

    @patch('pymongo.MongoClient')
    def test_insert_one(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.test_collection.insert_one.return_value.inserted_id = "123"
        result = self.db_manager.insert_one("test_collection", {"key": "value"})
        self.assertEqual(result, "123")
        mock_mongo.return_value.vial_mcp.test_collection.insert_one.assert_called_with({"key": "value"})

    @patch('pymongo.MongoClient')
    def test_find_one(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.test_collection.find_one.return_value = {"key": "value"}
        result = self.db_manager.find_one("test_collection", {"key": "value"})
        self.assertEqual(result, {"key": "value"})
        mock_mongo.return_value.vial_mcp.test_collection.find_one.assert_called_with({"key": "value"})

    @patch('pymongo.MongoClient')
    def test_find_many(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.test_collection.find.return_value.skip.return_value.limit.return_value = [{"key": "value"}]
        result = self.db_manager.find_many("test_collection", {"key": "value"}, limit=10, skip=0)
        self.assertEqual(result, [{"key": "value"}])
        mock_mongo.return_value.vial_mcp.test_collection.find.assert_called_with({"key": "value"})

    @patch('pymongo.MongoClient')
    def test_update_one(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.test_collection.update_one.return_value.modified_count = 1
        result = self.db_manager.update_one("test_collection", {"key": "value"}, {"new_key": "new_value"})
        self.assertEqual(result, 1)
        mock_mongo.return_value.vial_mcp.test_collection.update_one.assert_called_with({"key": "value"}, {"$set": {"new_key": "new_value"}})

    @patch('pymongo.MongoClient')
    def test_delete_one(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.test_collection.delete_one.return_value.deleted_count = 1
        result = self.db_manager.delete_one("test_collection", {"key": "value"})
        self.assertEqual(result, 1)
        mock_mongo.return_value.vial_mcp.test_collection.delete_one.assert_called_with({"key": "value"})

if __name__ == "__main__":
    unittest.main()
