# main/server/mcp/auth/test_auth_manager.py
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from .auth_manager import AuthManager

class TestAuthManager(unittest.TestCase):
    def setUp(self):
        self.auth_manager = AuthManager()

    @patch('main.server.mcp.db.db_manager.DBManager.insert_one')
    def test_create_token(self, mock_insert):
        mock_insert.return_value = "session_id"
        token = self.auth_manager.create_token("test_user", {"role": "admin"})
        self.assertIsNotNone(token)
        mock_insert.assert_called_once()
        payload = jwt.decode(token, self.auth_manager.secret_key, algorithms=["HS256"])
        self.assertEqual(payload["sub"], "test_user")
        self.assertEqual(payload["role"], "admin")

    @patch('main.server.mcp.db.db_manager.DBManager.find_one')
    def test_verify_token(self, mock_find):
        token = jwt.encode({
            "sub": "test_user",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }, self.auth_manager.secret_key, algorithm="HS256")
        mock_find.return_value = {"token": token, "expires_at": datetime.utcnow() + timedelta(minutes=30)}
        payload = self.auth_manager.verify_token(token)
        self.assertEqual(payload["sub"], "test_user")

    @patch('main.server.mcp.db.db_manager.DBManager.delete_one')
    def test_invalidate_token(self, mock_delete):
        mock_delete.return_value = 1
        result = self.auth_manager.invalidate_token("test_token")
        self.assertTrue(result)
        mock_delete.assert_called_with("sessions", {"token": "test_token"})

    @patch('main.server.mcp.db.db_manager.DBManager.update_one')
    def test_webauthn_challenge(self, mock_update):
        mock_update.return_value = 1
        challenge = self.auth_manager.webauthn_challenge("test_user")
        self.assertIn("challenge", challenge)
        self.assertEqual(len(challenge["challenge"]), 64)
        mock_update.assert_called_with("users", {"user_id": "test_user"}, {"webauthn_challenge": challenge["challenge"]})

if __name__ == "__main__":
    unittest.main()
