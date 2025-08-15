# main/server/mcp/utils/test_service_registry.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..api_gateway.service_registry import app, db_manager

class TestServiceRegistry(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    def test_register_service(self, mock_insert):
        mock_insert.return_value = "service123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/services",
                json={"name": "test_service", "url": "http://test", "health_check": "/health", "metadata": {"version": "1.0"}},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service_id"], "service123")
        self.assertEqual(response.json()["name"], "test_service")
        mock_insert.assert_called_once()

    @patch.object(db_manager, 'find_many')
    def test_list_services(self, mock_find):
        mock_find.return_value = [{"_id": "service123", "name": "test_service", "url": "http://test", "health_check": "/health", "metadata": {"version": "1.0"}, "timestamp": "2025-08-14T00:00:00"}]
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/services", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["name"], "test_service")

    @patch.object(db_manager, 'find_one')
    def test_get_service(self, mock_find):
        mock_find.return_value = {"_id": "service123", "name": "test_service", "url": "http://test", "health_check": "/health", "metadata": {"version": "1.0"}, "timestamp": "2025-08-14T00:00:00"}
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/services/test_service", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "test_service")
        mock_find.assert_called_with("services", {"name": "test_service"})

if __name__ == "__main__":
    unittest.main()
