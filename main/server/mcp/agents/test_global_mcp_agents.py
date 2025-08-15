# main/server/mcp/agents/test_global_mcp_agents.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .global_mcp_agents import app, db_manager, global_agents

class TestGlobalAgents(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch.object(db_manager, 'insert_one')
    @patch.object(global_agents, 'execute_task')
    def test_create_task(self, mock_execute_task, mock_insert):
        mock_insert.return_value = "task123"
        mock_execute_task.return_value = {"status": "completed", "output": "Processed test_task"}
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/agents/tasks",
                json={"task_id": "task123", "user_id": "test_user", "task_type": "test_task", "parameters": {"param": "value"}},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_id"], "task123")
        self.assertEqual(response.json()["result"]["status"], "completed")
        mock_insert.assert_called_once()
        mock_execute_task.assert_called_once()

    @patch.object(db_manager, 'find_many')
    def test_list_tasks(self, mock_find):
        mock_find.return_value = [
            {"_id": "task123", "task_id": "task123", "user_id": "test_user", "task_type": "test_task", "parameters": {"param": "value"}, "status": "completed", "result": {}, "timestamp": "2025-08-14T00:00:00"}
        ]
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/agents/tasks/test_user", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["task_id"], "task123")
        mock_find.assert_called_with("tasks", {"user_id": "test_user"})

if __name__ == "__main__":
    unittest.main()
