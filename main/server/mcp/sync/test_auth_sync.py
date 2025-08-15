# main/server/mcp/sync/test_auth_sync.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .auth_sync import app, db_manager

class TestAuthSync(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'find_one')
    @patch.object(db_manager, 'insert_one')
    @patch('requests.post')
    def test_sync_auth(self, mock_post, mock_insert, mock_find):
        mock_find.return_value = {"user_id": "test_user", "token": "test_token"}
        mock_insert.return_value = "sync123"
        mock_post.return_value.raise_for_status.return_value = None
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/sync/auth",
                json={"user_id": "test_user", "token": "test_token", "node_id": "node1"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        self.assertEqual(response.json()["node_id"], "node1")
        mock_insert.assert_called_once()
        mock_post.assert_called()

if __name__ == "__main__":
    unittest.main()
