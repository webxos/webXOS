import unittest
from unittest.mock import patch
from flask import Flask
from mock_backend import app
import json

class TestMockBackend(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['mongo'])
        self.assertEqual(data['version'], '2.8')
        self.assertIn('services', data)

    def test_auth_login(self):
        response = self.app.post('/api/auth/login', json={'userId': 'test-user'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('apiKey', data)
        self.assertIn('walletAddress', data)
        self.assertIn('walletHash', data)

    def test_auth_generate_api_key(self):
        response = self.app.post('/api/auth/api-key/generate', json={'userId': 'test-user'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('apiKey', data)
        self.assertIn('walletAddress', data)
        self.assertIn('walletHash', data)

    def test_log_error(self):
        response = self.app.post('/api/log_error', json={
            'error': 'Test error',
            'endpoint': '/api/test',
            'timestamp': '2025-08-13T17:00:00Z',
            'source': 'test',
            'rawResponse': 'Test response'
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'logged')

    def test_vial_prompt(self):
        response = self.app.post('/api/vials/vial1/prompt', json={'vialId': 'vial1', 'prompt': 'Test prompt', 'blockHash': 'test-hash'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['response'], 'Prompt processed for vial1')

    def test_vial_task(self):
        response = self.app.post('/api/vials/vial1/task', json={'vialId': 'vial1', 'task': 'Test task', 'blockHash': 'test-hash'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Task assigned to vial1')

    def test_vial_config(self):
        response = self.app.put('/api/vials/vial1/config', json={'vialId': 'vial1', 'key': 'model', 'value': 'gpt-3', 'blockHash': 'test-hash'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Config updated for vial1')

    def test_vials_void(self):
        response = self.app.delete('/api/vials/void')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'All vials reset')

    def test_wallet_create(self):
        response = self.app.post('/api/wallet/create', json={'userId': 'test-user', 'address': 'test-wallet', 'balance': 0, 'hash': 'test-hash', 'webxos': 0.0})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Wallet created')
        self.assertIn('address', data)

    def test_wallet_import(self):
        response = self.app.post('/api/wallet/import', json={'userId': 'test-user', 'address': 'test-wallet', 'hash': 'test-hash', 'webxos': 0.0})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Wallet imported')

    def test_wallet_transaction(self):
        self.app.post('/api/wallet/create', json={'userId': 'test-user', 'address': 'test-wallet', 'balance': 0, 'hash': 'test-hash', 'webxos': 0.0})
        response = self.app.post('/api/wallet/transaction', json={'userId': 'test-user', 'type': 'test'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Transaction recorded')

    def test_quantum_link(self):
        response = self.app.post('/api/quantum/link', json={'vials': ['vial1', 'vial2']})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('statuses', data)
        self.assertIn('latencies', data)

    def test_blockchain_transaction(self):
        response = self.app.post('/api/blockchain/transaction', json={'type': 'test', 'data': {}, 'timestamp': '2025-08-13T17:00:00Z', 'hash': 'test-hash'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Transaction recorded')

if __name__ == '__main__':
    unittest.main()
