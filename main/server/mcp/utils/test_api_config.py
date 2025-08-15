# main/server/mcp/utils/test_api_config.py
import unittest
from .api_config import APIConfig, EndpointConfig

class TestAPIConfig(unittest.TestCase):
    def setUp(self):
        self.api_config = APIConfig()

    def test_load_config(self):
        self.assertGreater(len(self.api_config.endpoints), 0)
        self.assertIn("auth", self.api_config.endpoints)
        self.assertEqual(self.api_config.endpoints["auth"].method, "POST")

    def test_get_endpoint(self):
        endpoint = self.api_config.get_endpoint("auth")
        self.assertIsInstance(endpoint, EndpointConfig)
        self.assertEqual(endpoint.name, "auth")
        with self.assertRaises(ValueError):
            self.api_config.get_endpoint("nonexistent")

    def test_list_endpoints(self):
        endpoints = self.api_config.list_endpoints()
        self.assertIsInstance(endpoints, list)
        self.assertGreater(len(endpoints), 0)
        self.assertTrue(all(isinstance(e, EndpointConfig) for e in endpoints))

if __name__ == "__main__":
    unittest.main()
