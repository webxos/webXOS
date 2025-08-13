import pytest
from fastapi.testclient import TestClient
from db.auth_sync import app, AuthSyncRequest
import jwt

client = TestClient(app)

@pytest.mark.asyncio
async def test_auth_sync():
    secret_key = "VIAL_MCP_SECRET_2025"
    token = jwt.encode({"user_id": "test_user"}, secret_key, algorithm="HS256")
    request = AuthSyncRequest(user_id="test_user", api_key=token, wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/auth_sync", json=request.dict())
    assert response.status_code == 200
    assert response.json()["status"] == "authenticated"
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "auth_sync"

@pytest.mark.asyncio
async def test_invalid_auth_sync():
    request = AuthSyncRequest(user_id="test_user", api_key="invalid_token", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/auth_sync", json=request.dict())
    assert response.status_code == 500
    assert "Auth sync error" in response.json()["detail"]
