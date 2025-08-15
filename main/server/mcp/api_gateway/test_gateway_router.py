# main/server/mcp/api_gateway/test_gateway_router.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
from ..api_gateway.gateway_router import route_request

client = TestClient(app)

def test_route_request():
    response = client.post("/mcp", json={"method": "test", "params": {}, "id": 1})
    assert response.status_code == 200
    assert "jsonrpc" in response.json()
