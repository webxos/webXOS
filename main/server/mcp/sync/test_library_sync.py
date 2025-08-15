# main/server/mcp/sync/test_library_sync.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .library_sync import app, db_manager

class TestLibrarySync(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'find_one')
    @patch.object(db_manager, 'insert_one')
    @patch.object(db_manager, 'update_one')
    @patch('requests.post')
    def test_sync_library(self, mock_post, mock_update, mock_insert, mock_find):
        mock_find.return_value = None
        mock_insert.return_value = "item123"
        mock_post.return_value.raise_for_status.return_value = None
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/sync/library",
                json={
                    "user_id": "test_user",
                    "item_id": "item123",
                    "node_id": "node1",
                    "item_data": {"title": "Test Resource", "content": "Sample content"}
                },
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        self.assertEqual(response.json()["item_id"], "item123")
        mock_insert.assert_called()
        mock_post.assert_called()

if __name__ == "__main__":
    unittest.main()
