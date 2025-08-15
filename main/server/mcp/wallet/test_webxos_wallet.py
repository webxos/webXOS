# main/server/mcp/wallet/test_webxos_wallet.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .webxos_wallet import app
from datetime import datetime

class TestMCPWalletServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('pymongo.MongoClient')
    @patch('web3.Web3')
    def test_verify_wallet(self, mock_web3, mock_mongo):
        mock_web3.return_value.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
        mock_web3.return_value.is_address.return_value = True
        mock_mongo.return_value.vial_mcp.wallets.update_one.return_value = None
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.get(
                "/wallet/verify?user_id=test_user&address=0x1234567890abcdef1234567890abcdef12345678",
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        self.assertEqual(response.json()["balance"], 1.0)

    @patch('pymongo.MongoClient')
    @patch('web3.Web3')
    def test_transact(self, mock_web3, mock_mongo):
        mock_web3.return_value.is_address.return_value = True
        mock_mongo.return_value.vial_mcp.wallets.find_one.return_value = {"user_id": "test_user", "address": "0x1234567890abcdef1234567890abcdef12345678"}
        mock_mongo.return_value.vial_mcp.wallets.insert_one.return_value = None
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.post(
                "/wallet/transact",
                json={"user_id": "test_user", "to_address": "0xabcdef1234567890abcdef1234567890abcdef12", "amount": 0.1, "currency": "ETH"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

if __name__ == "__main__":
    unittest.main()
