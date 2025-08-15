# main/server/mcp/tests/test_e2e_wallet.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..wallet.webxos_wallet import app

class TestE2EWallet(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.find_one')
    def test_get_wallet_balance(self, mock_find, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_find.return_value = {"user_id": "test_user", "address": "0x123", "balance": "10.5 ETH"}
        token = "mock_token"
        response = self.client.get(
            "/wallet/balance/test_user",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["balance"], "10.5 ETH")
        mock_find.assert_called_with("wallets", {"user_id": "test_user"})

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.insert_one')
    def test_create_wallet(self, mock_insert, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_insert.return_value = "wallet123"
        token = "mock_token"
        response = self.client.post(
            "/wallet",
            json={"user_id": "test_user", "address": "0x123"},
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["wallet_id"], "wallet123")
        mock_insert.assert_called_once()

if __name__ == "__main__":
    unittest.main()
