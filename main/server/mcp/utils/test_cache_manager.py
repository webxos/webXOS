# main/server/mcp/utils/test_cache_manager.py
import unittest
from unittest.mock import patch
import json
from .cache_manager import CacheManager
import time

class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.cache_manager = CacheManager()

    @patch('redis.Redis')
    def test_set_and_get_redis(self, mock_redis):
        mock_redis.return_value.setex.return_value = True
        mock_redis.return_value.get.return_value = json.dumps({"data": "test"})
        self.cache_manager.redis_client = mock_redis.return_value
        result = self.cache_manager.set("test_key", {"data": "test"}, expiry=60)
        self.assertTrue(result)
        value = self.cache_manager.get("test_key")
        self.assertEqual(value, {"data": "test"})
        mock_redis.return_value.setex.assert_called_with("test_key", 60, json.dumps({"data": "test"}))
        mock_redis.return_value.get.assert_called_with("test_key")

    def test_set_and_get_in_memory(self):
        self.cache_manager.redis_client = None
        result = self.cache_manager.set("test_key", {"data": "test"}, expiry=60)
        self.assertTrue(result)
        value = self.cache_manager.get("test_key")
        self.assertEqual(value, {"data": "test"})
        self.assertIn("test_key", self.cache_manager.in_memory_cache)

    @patch('redis.Redis')
    def test_delete_redis(self, mock_redis):
        mock_redis.return_value.delete.return_value = 1
        self.cache_manager.redis_client = mock_redis.return_value
        result = self.cache_manager.delete("test_key")
        self.assertTrue(result)
        mock_redis.return_value.delete.assert_called_with("test_key")

    def test_delete_in_memory(self):
        self.cache_manager.redis_client = None
        self.cache_manager.in_memory_cache["test_key"] = {"value": {"data": "test"}, "expiry": int(time.time()) + 60}
        result = self.cache_manager.delete("test_key")
        self.assertTrue(result)
        self.assertNotIn("test_key", self.cache_manager.in_memory_cache)

    def test_get_expired_in_memory(self):
        self.cache_manager.redis_client = None
        self.cache_manager.in_memory_cache["test_key"] = {"value": {"data": "test"}, "expiry": int(time.time()) - 10}
        value = self.cache_manager.get("test_key")
        self.assertIsNone(value)
        self.assertNotIn("test_key", self.cache_manager.in_memory_cache)

if __name__ == "__main__":
    unittest.main()
