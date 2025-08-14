# main/server/mcp/resources/test_mcp_server_resources.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .mcp_server_resources import app
from datetime import datetime

class TestMCPResourcesServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('pymongo.MongoClient')
    @patch('psutil.cpu_percent', return_value=50.0)
    @patch('psutil.virtual_memory', return_value=type('obj', (), {'percent': 60.0}))
    @patch('psutil.disk_usage', return_value=type('obj', (), {'percent': 70.0}))
    def test_get_resources(self, mock_disk, mock_memory, mock_cpu, mock_mongo):
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.get("/resources", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        resources = response.json()
        self.assertEqual(len(resources), 3)
        self.assertEqual(resources[0]["type"], "CPU")
        self.assertEqual(resources[0]["usage"], 50.0)
        self.assertEqual(resources[1]["type"], "Memory")
        self.assertEqual(resources[2]["type"], "Disk")

    @patch('pymongo.MongoClient')
    def test_get_resources_history(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.resources.find.return_value.sort.return_value.limit.return_value = [
            {"timestamp": datetime.utcnow(), "resources": [{"type": "CPU", "usage": 50.0, "total": 100.0, "unit": "%"}]}
        ]
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.get("/resources/history", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["resources"][0]["type"], "CPU")

if __name__ == "__main__":
    unittest.main()
