import pytest
from fastapi.testclient import TestClient
from vial2.mcp.api import auth_endpoint

client = TestClient(auth_endpoint.app)

def test_auth_login():
    response = client.post("/mcp/api/auth/login", json={"provider": "github"})
    assert response.status_code == 200
    assert "url" in response.json()

def test_auth_logout():
    response = client.post("/mcp/api/auth/logout", json={"token": "test_token"})
    assert response.status_code == 200
    assert response.json() == {"message": "Logged out"}