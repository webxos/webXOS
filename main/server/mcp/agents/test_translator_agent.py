# main/server/mcp/agents/test_translator_agent.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .translator_agent import app, db_manager

class TestTranslatorAgent(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('requests.post')
    @patch.object(db_manager, 'insert_one')
    def test_translate_text(self, mock_insert, mock_post):
        mock_post.return_value.json.return_value = {"translated_text": "Hola"}
        mock_post.return_value.raise_for_status.return_value = None
        mock_insert.return_value = "trans123"
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.post(
                "/agents/translate",
                json={"user_id": "test_user", "text": "Hello", "source_lang": "en", "target_lang": "es"},
                headers={"Authorization": f"Bearer {token}"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["translation_id"], "trans123")
        self.assertEqual(response.json()["translated_text"], "Hola")
        self.assertEqual(response.json()["target_lang"], "es")
        mock_insert.assert_called_once()

if __name__ == "__main__":
    unittest.main()
