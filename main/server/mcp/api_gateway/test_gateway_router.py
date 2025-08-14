# main/server/mcp/api_gateway/test_gateway_router.py
import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from .gateway_router import app, SERVICE_REGISTRY
import httpx

class TestGatewayRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('httpx.AsyncClient')
    async def test_route_request(self, mock_client):
        mock_client.return_value.__aenter__.return_value.request.return_value.json.return_value = {"status": "success"}
        response = self.client.post("/route", json={
            "service": "auth",
            "endpoint": "token",
            "method": "POST",
            "data": {"user_id": "test_user"}
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success"})
        mock_client.return_value.__aenter__.return_value.request.assert_called_with(
            method="POST",
            url=f"{SERVICE_REGISTRY['auth']}/token",
            json={"user_id": "test_user"},
            params=None
        )

    @patch('httpx.AsyncClient')
    async def test_health_check(self, mock_client):
        mock_client.return_value.__aenter__.return_value.get.return_value.json.return_value = {"status": "healthy"}
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertEqual(len(response.json()["services"]), len(SERVICE_REGISTRY))

    def test_rate_limit_exceeded(self):
        with patch('main.server.mcp.utils.rate_limiter.RateLimiter.allow', return_value=False):
            response = self.client.post("/route", json={
                "service": "auth",
                "endpoint": "token",
                "method": "POST",
                "data": {}
            })
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.json()["detail"], "Rate limit exceeded")

if __name__ == "__main__":
    unittest.main()
