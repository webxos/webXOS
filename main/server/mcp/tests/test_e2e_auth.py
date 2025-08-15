# main/server/mcp/tests/test_e2e_auth.py
import pytest
from fastapi.testclient import TestClient
from ..utils.api_config import APIConfig
from ..utils.auth_manager import AuthManager
from ..utils.cache_manager import CacheManager
import json

@pytest.fixture
def client():
    from fastapi import FastAPI
    from ..api_gateway.gateway_router import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
async def auth_manager():
    manager = AuthManager({"address": "0x123", "hash": "abc123", "reputation": 1000})
    yield manager

@pytest.fixture
async def cache_manager():
    manager = CacheManager()
    yield manager
    await manager.redis_client.flushdb()
    await manager.close()

@pytest.mark.asyncio
async def test_create_session(client, auth_manager, cache_manager):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.createSession",
            "params": {"user_id": "test_user", "mfa_verified": False},
            "id": 1
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "session_id" in data["result"]
    assert "access_token" in data["result"]

@pytest.mark.asyncio
async def test_initiate_mfa(client, auth_manager):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.initiateMFA",
            "params": {"user_id": "test_user", "mfa_method": "email"},
            "id": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "mfa_token" in data["result"]
    assert data["result"]["method"] == "email"

@pytest.mark.asyncio
async def test_verify_mfa(client, auth_manager, cache_manager, mocker):
    mocker.patch.object(auth_manager, "validate_wallet", return_value=True)
    mfa_token = await cache_manager.set_cache(
        "mfa:test_user", {"mfa_token": "test_mfa_token", "mfa_code": "123456"}
    )
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.verifyMFA",
            "params": {"user_id": "test_user", "mfa_token": "test_mfa_token", "mfa_code": "123456"},
            "id": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["verified"] is True

@pytest.mark.asyncio
async def test_revoke_session(client, auth_manager, cache_manager):
    session_id = await cache_manager.set_cache(
        "session:test_user", {"session_id": "test_session", "user_id": "test_user"}
    )
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.revokeSession",
            "params": {"session_id": "test_session", "user_id": "test_user"},
            "id": 4
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["status"] == "revoked"
    assert await cache_manager.get_cache("session:test_user") is None
