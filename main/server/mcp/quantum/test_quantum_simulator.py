# main/server/mcp/quantum/test_quantum_simulator.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .quantum_simulator import app, db_manager
from datetime import datetime

class TestQuantumSimulator(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('qiskit.execute')
    @patch.object(db_manager, 'insert_one')
    def test_simulate_circuit(self, mock_insert, mock_execute):
        mock_execute.return_value.result.return_value.to_dict.return_value = {"counts": {"00": 512, "11": 512}}
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/quantum/simulate",
                json={
                    "vial_id": "vial1",
                    "circuit_data": {"qasm": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;"},
                    "user_id": "test_user"
                },
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["vial_id"], "vial1")
        self.assertEqual(response.json()["result"]["counts"], {"00": 512, "11": 512})
        mock_insert.assert_called_once()

if __name__ == "__main__":
    unittest.main()
