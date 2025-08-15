# main/server/mcp/utils/test_api_docs.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .api_docs import app, APIDocs

class TestAPIDocs(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.api_docs = APIDocs()

    @patch('main.server.mcp.utils.api_config.APIConfig.list_endpoints')
    def test_generate_openapi(self, mock_list_endpoints):
        mock_list_endpoints.return_value = [
            type("Endpoint", (), {"name": "test", "url": "http://test", "method": "GET", "requires_auth": True})
        ]
        schema = self.api_docs.generate_openapi(app)
        self.assertEqual(schema["info"]["title"], "Vial MCP API")
        self.assertIn("/test", schema["paths"])
        self.assertEqual(schema["paths"]["/test"]["get"]["summary"], "Access test service")

    @patch('yaml.dump')
    def test_save_openapi_yaml(self, mock_yaml_dump):
        schema = {"info": {"title": "Test API"}}
        self.api_docs.save_openapi_yaml(schema)
        mock_yaml_dump.assert_called_with(schema, unittest.mock.ANY, sort_keys=False)

    @patch('main.server.mcp.utils.api_docs.APIDocs.generate_openapi')
    def test_get_openapi_docs(self, mock_generate):
        mock_generate.return_value = {"info": {"title": "Vial MCP API"}}
        token = "mock_token"
        with patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.verify_token', return_value={"sub": "test_user"}):
            response = self.client.get("/docs/openapi", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["info"]["title"], "Vial MCP API")

if __name__ == "__main__":
    unittest.main()
