# main/server/mcp/tests/test_e2e_auth.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..auth.mcp_server_auth import app
import jwt
from datetime import datetime, timedelta

class TestE2EAuth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main.server.mcp.db.db_manager.DBManager.find_one')
    @patch('main.server.mcp.db.db_manager.DBManager.insert_one')
    def test_login_success(self, mock_insert, mock_find):
        mock_find.return_value = {"user_id": "test_user", "username": "test", "password_hash": "hashed_password"}
        mock_insert.return_value = "session123"
        response = self.client.post(
            "/auth/token",
            json={"username": "test", "password": "password"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("token", response.json())
        self.assertEqual(response.json()["userId"], "test_user")
        mock_insert.assert_called_once()

    @patch('main.server.mcp.db.db_manager.DBManager.find_one')
    def test_webauthn_registration(self, mock_find):
        mock_find.return_value = {"user_id": "test_user"}
        response = self.client.post(
            "/auth/webauthn/register",
            json={"user_id": "test_user"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("publicKey", response.json())
        mock_find.assert_called_with("users", {"user_id": "test_user"})

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    def test_logout(self, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        token = jwt.encode(
            {"sub": "test_user", "iat": datetime.utcnow(), "exp": datetime.utcnow() + timedelta(minutes=30)},
            "secret_key",
            algorithm="HS256"
        )
        response = self.client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success", "message": "Logged out"})

if __name__ == "__main__":
    unittest.main()
