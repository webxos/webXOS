# main/server/mcp/utils/test_health_check.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .health_check import app, HealthStatus

class TestHealthCheck(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(HealthStatus, 'check_mongo')
    @patch.object(HealthStatus, 'check_redis')
    def test_health_check_healthy(self, mock_check_redis, mock_check_mongo):
        mock_check_mongo.return_value = True
        mock_check_redis.return_value = True
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["overall"])
        self.assertTrue(response.json()["mongo"])
        self.assertTrue(response.json()["redis"])
        self.assertIn("timestamp", response.json())

    @patch.object(HealthStatus, 'check_mongo')
    @patch.object(HealthStatus, 'check_redis')
    def test_health_check_unhealthy_mongo(self, mock_check_redis, mock_check_mongo):
        mock_check_mongo.return_value = False
        mock_check_redis.return_value = True
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "One or more services are unhealthy")

if __name__ == "__main__":
    unittest.main()
