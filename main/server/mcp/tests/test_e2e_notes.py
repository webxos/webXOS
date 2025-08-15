# main/server/mcp/tests/test_e2e_notes.py
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
async def test_create_note(client, auth_manager, cache_manager):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.createNote",
            "params": {
                "user_id": "test_user",
                "title": "Test Note",
                "content": "This is a test note",
                "tags": ["test", "note"]
            },
            "id": 1
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "note_id" in data["result"]
    assert data["result"]["status"] == "created"

@pytest.mark.asyncio
async def test_add_sub_note(client, auth_manager, cache_manager):
    # First create a parent note
    parent_response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.createNote",
            "params": {
                "user_id": "test_user",
                "title": "Parent Note",
                "content": "This is a parent note",
                "tags": ["parent"]
            },
            "id": 2
        },
        headers={"Authorization": "Bearer test_token"}
    )
    parent_note_id = parent_response.json()["result"]["note_id"]
    
    # Add sub-note
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.addSubNote",
            "params": {
                "parent_note_id": parent_note_id,
                "content": "This is a sub-note",
                "user_id": "test_user"
            },
            "id": 3
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "sub_note_id" in data["result"]
    assert data["result"]["status"] == "created"

@pytest.mark.asyncio
async def test_get_notes(client, auth_manager, cache_manager):
    # Create a note first
    client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.createNote",
            "params": {
                "user_id": "test_user",
                "title": "Test Note",
                "content": "This is a test note",
                "tags": ["test"]
            },
            "id": 4
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.getNotes",
            "params": {"user_id": "test_user", "tags": ["test"]},
            "id": 5
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert len(data["result"]) >= 1
    assert data["result"][0]["title"] == "Test Note"
