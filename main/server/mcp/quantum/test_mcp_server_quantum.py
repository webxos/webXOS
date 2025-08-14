# main/server/mcp/quantum/test_mcp_server_quantum.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .mcp_server_quantum import app
from datetime import datetime

class TestMCPQuantumServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('pymongo.MongoClient')
    @patch('qiskit.execute')
    def test_execute_quantum_circuit(self, mock_execute, mock_mongo):
        mock_execute.return_value.result.return_value.get_counts.return_value = {"00": 512, "11": 512}
        mock_mongo.return_value.vial_mcp.quantum_circuits.insert_one.return_value = None
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.post(
                "/quantum/execute",
                json={"vial_id": "vial1", "qubits": 2, "circuit": "h(0);cx(0,1);measure_all()"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vial_id"], "vial1")
        self.assertIn("result", response.json())
        self.assertIn("execution_time", response.json())

    @patch('pymongo.MongoClient')
    def test_get_quantum_history(self, mock_mongo):
        mock_mongo.return_value.vial_mcp.quantum_circuits.find.return_value.sort.return_value.limit.return_value = [
            {"vial_id": "vial1", "result": {"00": 512}, "execution_time": 0.1, "timestamp": datetime.utcnow()}
        ]
        token = "mock_token"
        with patch.object(app.dependency_overrides.get("oauth2_scheme"), "verify_token", return_value={"sub": "test_user"}):
            response = self.client.get("/quantum/history/vial1", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["vial_id"], "vial1")

if __name__ == "__main__":
    unittest.main()
