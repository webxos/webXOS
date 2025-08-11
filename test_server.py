import pytest
from fastapi.testclient import TestClient
from vial.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_auth_endpoint():
    response = client.post("/auth", json={"client": "test", "deviceId": "test", "sessionId": "test", "networkId": "test"})
    assert response.status_code == 200
    assert "token" in response.json()
    assert "address" in response.json()

def test_stream_endpoint():
    response = client.post("/auth", json={"client": "test", "deviceId": "test", "sessionId": "test", "networkId": "test"})
    token = response.json()["token"]
    response = client.get("/stream/test", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]