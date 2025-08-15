# main/server/mcp/agents/test_global_mcp_agents.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .global_mcp_agents import app, db_manager
from datetime import datetime

class TestGlobalMCPAgents(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    def test_create_task(self, mock_insert):
        mock_insert.return_value = "123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/agents/tasks",
                json={"task_id": "task1", "type": "quantum", "parameters": {"qubits": 2}, "user_id": "test_user"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_id"], "task1")
        self.assertEqual(response.json()["user_id"], "test_user")

    @patch.object(db_manager, 'find_many')
    def test_get_tasks(self, mock_find):
        mock_find.return_value = [
            {"task_id": "task1", "type": "quantum", "status": "pending", "parameters": {"qubits": 2}, "user_id": "test_user", "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
        ]
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/agents/tasks/test_user", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["task_id"], "task1")

    @patch.object(db_manager, 'find_one')
    @patch.object(db_manager, 'update_one')
    def test_execute_task(self, mock_update, mock_find):
        mock_find.return_value = {"task_id": "task1", "type": "quantum", "status": "pending", "parameters": {"qubits": 2}, "user_id": "test_user"}
        mock_update.return_value = 1
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post("/agents/tasks/task1/execute", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

if __name__ == "__main__":
    unittest.main()
