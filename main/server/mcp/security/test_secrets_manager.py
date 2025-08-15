# main/server/mcp/security/test_secrets_manager.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .secrets_manager import app, db_manager, secrets_manager

class TestSecretsManager(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    @patch.object(secrets_manager, 'encrypt_secret')
    def test_store_secret(self, mock_encrypt, mock_insert):
        mock_encrypt.return_value = "encrypted_value"
        mock_insert.return_value = "secret123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/secrets",
                json={"user_id": "test_user", "key": "api_key", "value": "sensitive_data"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        self.assertEqual(response.json()["key"], "api_key")
        mock_insert.assert_called_once()
        mock_encrypt.assert_called_with("sensitive_data")

    @patch.object(db_manager, 'find_one')
    def test_retrieve_secret(self, mock_find):
        mock_find.return_value = {"user_id": "test_user", "key": "api_key", "value": "encrypted_value", "timestamp": "2025-08-14T00:00:00"}
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get(
                "/secrets/test_user/api_key",
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        self.assertEqual(response.json()["key"], "api_key")
        mock_find.assert_called_with("secrets", {"user_id": "test_user", "key": "api_key"})

if __name__ == "__main__":
    unittest.main()
