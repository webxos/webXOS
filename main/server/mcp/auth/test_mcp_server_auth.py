# main/server/mcp/auth/test_mcp_server_auth.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .mcp_server_auth import app
from datetime import datetime

class TestMCPAuthServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('pymongo.MongoClient')
    @patch('oci.config.from_file')
    @patch('oci.auth.signers.InstancePrincipalsSecurityTokenSigner')
    def test_login_for_access_token(self, mock_signer, mock_oci_config, mock_mongo):
        mock_mongo.return_value.vial_mcp.users.find_one.return_value = None
        response = self.client.post("/token", json={"user_id": "test_user", "email": "test@example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertEqual(response.json()["token_type"], "bearer")

    @patch('pymongo.MongoClient')
    @patch('webauthn.generate_registration_options')
    def test_webauthn_register(self, mock_webauthn, mock_mongo):
        mock_webauthn.return_value = type('Options', (), {'challenge': b'123', '__dict__': {'challenge': b'123'}})
        response = self.client.post("/webauthn/register", json={"user_id": "test_user", "email": "test@example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("challenge", response.json())

    @patch('pymongo.MongoClient')
    @patch('webauthn.verify_registration_response')
    def test_webauthn_verify(self, mock_verify, mock_mongo):
        mock_mongo.return_value.vial_mcp.users.find_one.return_value = {"user_id": "test_user", "webauthn_challenge": "123"}
        mock_verify.return_value = type('Verified', (), {'credential_id': b'456'})
        response = self.client.post("/webauthn/verify?user_id=test_user", json={"credential": {}, "client_data": "", "attestation_object": ""})
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())

    @patch('pymongo.MongoClient')
    def test_read_users_me(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.users.find_one.return_value = {"user_id": "test_user", "last_login": datetime.utcnow(), "email": "test@example.com"}
        token_response = self.client.post("/token", json={"user_id": "test_user", "email": "test@example.com"})
        token = token_response.json()["access_token"]
        response = self.client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")

if __name__ == "__main__":
    unittest.main()
