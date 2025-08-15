# main/server/mcp/tests/test_e2e_resources.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..resources.mcp_server_resources import app

class TestE2EResources(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    def test_get_system_metrics(self, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        token = "mock_token"
        with patch('psutil.cpu_percent', return_value=50.0), patch('psutil.virtual_memory', return_value=type('obj', (), {'percent': 75.0})):
            response = self.client.get(
                "/resources/metrics",
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cpu_usage"], 50.0)
        self.assertEqual(response.json()["memory_usage"], 75.0)

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.find_many')
    def test_list_resources(self, mock_find, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_find.return_value = [
            {"_id": "resource123", "user_id": "test_user", "title": "Test Resource", "content": "Sample content", "category": "docs", "timestamp": "2025-08-14T00:00:00"}
        ]
        token = "mock_token"
        response = self.client.get(
            "/resources?category=docs",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["title"], "Test Resource")
        mock_find.assert_called_with("resources", {"category": "docs"})

if __name__ == "__main__":
    unittest.main()
