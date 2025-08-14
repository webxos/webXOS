# main/server/mcp/wallet/test_webxos_wallet.py
import unittest
from unittest.mock import patch
from .webxos_wallet import WebXOSWallet
from datetime import datetime

class TestWebXOSWallet(unittest.TestCase):
    def setUp(self):
        self.wallet_manager = WebXOSWallet()

    @patch('pymongo.MongoClient')
    @patch('web3.Web3')
    async def test_create_wallet(self, mock_web3, mock_mongo):
        mock_web3.eth.account.create.return_value = type('Account', (), {'address': '0x123', 'privateKey': 'key'})
        mock_web3.keccak.return_value.hex.return_value = '0xhash'
        wallet = await self.wallet_manager.create_wallet("test_user")
        self.assertEqual(wallet["user_id"], "test_user")
        self.assertEqual(wallet["address"], "0x123")
        self.assertEqual(wallet["hash"], "0xhash")
        self.assertEqual(len(wallet["transactions"]), 1)
        self.assertEqual(wallet["transactions"][0]["type"], "created")

    @patch('pymongo.MongoClient')
    async def test_import_wallet(self, mock_mongo):
        wallet_data = """
## Wallet
- Address: 0x123
- Balance: 10.0 $WEBXOS
- Hash: 0xhash
- Transactions: []
"""
        wallet = await self.wallet_manager.import_wallet("test_user", wallet_data)
        self.assertEqual(wallet["address"], "0x123")
        self.assertEqual(wallet["webxos"], 10.0)
        self.assertEqual(len(wallet["transactions"]), 1)
        self.assertEqual(wallet["transactions"][0]["type"], "imported")

if __name__ == "__main__":
    unittest.main()
