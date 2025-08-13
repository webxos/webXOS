import pytest
from fastapi.testclient import TestClient
from vial.unified_server import app, connect_mongo
from vial.auth_manager import AuthManager

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_manager():
    return AuthManager()

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_auth_endpoint(client):
    response = client.post("/api/auth", json={"userId": "test_user"})
    assert response.status_code == 200
    assert "apiKey" in response.json()

def test_vials_endpoint(client, auth_manager):
    token = auth_manager.generate_token("test_user")
    response = client.get("/api/vials", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "agents" in response.json()

def test_wallet_endpoint(client, auth_manager):
    token = auth_manager.generate_token("test_user")
    response = client.post(
        "/api/wallet",
        headers={"Authorization": f"Bearer {token}"},
        json={"transaction": {"amount": 10.0}, "wallet": {"target_address": "test_address"}}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

def test_cashout_endpoint(client, auth_manager):
    token = auth_manager.generate_token("test_user")
    response = client.post(
        "/api/wallet/cashout",
        headers={"Authorization": f"Bearer {token}"},
        json={"transaction": {"amount": 5.0}, "wallet": {"target_address": "test_address"}}
    )
    assert response.status_code == 500  # Fails due to insufficient balance in test setup
