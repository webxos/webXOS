# main/server/mcp/agents/test_library_agent.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .library_agent import app, db_manager

class TestLibraryAgent(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    def test_add_library_item(self, mock_insert):
        mock_insert.return_value = "item123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/agents/library",
                json={"user_id": "test_user", "title": "Test Resource", "content": "Sample content", "tags": ["test"], "category": "docs"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["item_id"], "item123")
        self.assertEqual(response.json()["title"], "Test Resource")
        mock_insert.assert_called_once()

    @patch.object(db_manager, 'find_many')
    def test_get_library_items(self, mock_find):
        mock_find.return_value = [
            {"_id": "item123", "user_id": "test_user", "title": "Test Resource", "content": "Sample content", "tags": ["test"], "category": "docs", "timestamp": "2025-08-14T00:00:00"}
        ]
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/agents/library/test_user", headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["item_id"], "item123")

    @patch.object(db_manager, 'delete_one')
    def test_delete_library_item(self, mock_delete):
        mock_delete.return_value = 1
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.delete("/agents/library/item123", headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        mock_delete.assert_called_with("library", {"_id": "item123"})

if __name__ == "__main__":
    unittest.main()
