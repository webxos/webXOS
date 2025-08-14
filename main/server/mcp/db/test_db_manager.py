# server/mcp/db/test_db_manager.py
import unittest
import os
from .db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_vial_mcp.db"
        self.db_manager = DatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_save_and_get_agent_status(self):
        self.db_manager.save_agent_status("agent1", "active")
        status = self.db_manager.get_agent_status("agent1")
        self.assertEqual(status, "active")

    def test_save_task(self):
        self.db_manager.save_agent_status("agent1", "active")
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM agents WHERE agent_name = ?", ("agent1",))
            agent_id = cursor.fetchone()[0]
        task_id = self.db_manager.save_task(agent_id, "search_docs", "query: test", "pending")
        self.assertIsNotNone(task_id)

if __name__ == "__main__":
    unittest.main()
