# main/server/mcp/tests/test_e2e_notes.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..notes.mcp_server_notes import app

class TestE2ENotes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.insert_one')
    def test_create_note(self, mock_insert, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_insert.return_value = "note123"
        token = "mock_token"
        response = self.client.post(
            "/notes",
            json={"user_id": "test_user", "title": "Test Note", "content": "Sample content", "tags": ["test"]},
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["note_id"], "note123")
        self.assertEqual(response.json()["title"], "Test Note")
        mock_insert.assert_called_once()

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.find_many')
    def test_get_notes(self, mock_find, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_find.return_value = [
            {"_id": "note123", "user_id": "test_user", "title": "Test Note", "content": "Sample content", "tags": ["test"], "timestamp": "2025-08-14T00:00:00"}
        ]
        token = "mock_token"
        response = self.client.get(
            "/notes/test_user",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["title"], "Test Note")
        mock_find.assert_called_with("notes", {"user_id": "test_user"})

if __name__ == "__main__":
    unittest.main()
