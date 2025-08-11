import pytest
from fastapi.testclient import TestClient
from vial.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_auth_endpoint():
    response = client.post("/auth", json={
        "client": "vial",
        "deviceId": "test-device",
        "sessionId": "test-session",
        "networkId": "test-network"
    })
    assert response.status_code == 200
    assert "token" in response.json()
    assert "address" in response.json()

def test_train_endpoint_unauthorized():
    response = client.post("/train", data={"networkId": "test-network"})
    assert response.status_code == 401
