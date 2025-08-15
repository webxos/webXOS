# main/server/mcp/utils/test_webhook_manager.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .webhook_manager import app, db_manager

class TestWebhookManager(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    def test_register_webhook(self, mock_insert):
        mock_insert.return_value = "webhook123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/webhooks",
                json={"user_id": "test_user", "url": "http://example.com/webhook", "event_type": "vial_update"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["webhook_id"], "webhook123")
        self.assertEqual(response.json()["event_type"], "vial_update")
        mock_insert.assert_called_once()

    @patch.object(db_manager, 'find_many')
    @patch.object(db_manager, 'insert_one')
    @patch('requests.post')
    def test_notify_webhooks(self, mock_post, mock_insert, mock_find):
        mock_find.return_value = [{"_id": "webhook123", "url": "http://example.com/webhook", "event_type": "vial_update"}]
        mock_post.return_value.raise_for_status.return_value = None
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/webhooks/notify/vial_update",
                json={"data": "test_event"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        mock_find.assert_called_with("webhooks", {"event_type": "vial_update"})
        mock_post.assert_called_with("http://example.com/webhook", json={"data": "test_event"}, timeout=5)

if __name__ == "__main__":
    unittest.main()
