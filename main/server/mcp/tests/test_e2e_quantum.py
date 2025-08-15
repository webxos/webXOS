# main/server/mcp/tests/test_e2e_quantum.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..quantum.mcp_server_quantum import app

class TestE2EQuantum(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.insert_one')
    def test_simulate_quantum_circuit(self, mock_insert, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_insert.return_value = "result123"
        token = "mock_token"
        response = self.client.post(
            "/quantum/simulate",
            json={"vial_id": "vial123", "circuit_data": {"gates": ["H", "CNOT"]}, "user_id": "test_user"},
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("result_id", response.json())
        mock_insert.assert_called_once()

    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token')
    @patch('main.server.mcp.db.db_manager.DBManager.find_one')
    def test_get_quantum_results(self, mock_find, mock_verify_token):
        mock_verify_token.return_value = {"sub": "test_user"}
        mock_find.return_value = {"result_id": "result123", "results": {"state": "|00>"}, "vial_id": "vial123"}
        token = "mock_token"
        response = self.client.get(
            "/quantum/results/vial123",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"]["state"], "|00>")
        mock_find.assert_called_with("quantum_results", {"vial_id": "vial123"})

if __name__ == "__main__":
    unittest.main()
