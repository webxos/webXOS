import pytest
from fastapi.testclient import TestClient
from vial2.mcp.api import health_endpoint

client = TestClient(health_endpoint.app)

def test_health_check():
    response = client.get("/mcp/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"