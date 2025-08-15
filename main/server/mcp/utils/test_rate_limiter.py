# main/server/mcp/utils/test_rate_limiter.py
import unittest
from unittest.mock import patch
from .rate_limiter import RateLimiter

class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.rate_limiter = RateLimiter()

    @patch('redis.Redis')
    def test_check_rate_limit_redis(self, mock_redis):
        mock_redis.return_value.incr.return_value = 1
        mock_redis.return_value.expire.return_value = None
        result = self.rate_limiter.check_rate_limit("test_user")
        self.assertTrue(result)
        mock_redis.return_value.incr.assert_called_with("rate_limit:test_user:0")
        mock_redis.return_value.expire.assert_called_with("rate_limit:test_user:0", 60)

    def test_check_rate_limit_in_memory(self):
        self.rate_limiter.redis_client = None
        result = self.rate_limiter.check_rate_limit("test_user")
        self.assertTrue(result)
        self.assertEqual(self.rate_limiter.in_memory_cache["rate_limit:test_user:0"]["count"], 1)

    @patch('redis.Redis')
    def test_reset_rate_limit_redis(self, mock_redis):
        mock_redis.return_value.keys.return_value = ["rate_limit:test_user:0"]
        mock_redis.return_value.delete.return_value = 1
        result = self.rate_limiter.reset_rate_limit("test_user")
        self.assertTrue(result)
        mock_redis.return_value.delete.assert_called_with("rate_limit:test_user:0")

    def test_reset_rate_limit_in_memory(self):
        self.rate_limiter.redis_client = None
        self.rate_limiter.in_memory_cache["rate_limit:test_user:0"] = {"count": 1, "expiry": int(time.time()) + 60}
        result = self.rate_limiter.reset_rate_limit("test_user")
        self.assertTrue(result)
        self.assertNotIn("rate_limit:test_user:0", self.rate_limiter.in_memory_cache)

if __name__ == "__main__":
    unittest.main()
