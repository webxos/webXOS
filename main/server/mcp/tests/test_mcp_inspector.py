# main/server/mcp/tests/test_mcp_inspector.py
import pytest
from fastapi.testclient import TestClient
from ..unified_server import app
import json

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_initialize_request(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.initialize",
            "params": {"user_id": "test_user"},
            "id": 1
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["status"] == "initialized"

@pytest.mark.asyncio
async def test_list_tools_request(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.listTools",
            "params": {"user_id": "test_user"},
            "id": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "tools" in data["result"]
    assert len(data["result"]["tools"]) == 4

@pytest.mark.asyncio
async def test_call_tool_request(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.callTool",
            "params": {"user_id": "test_user", "tool_name": "wallet"},
            "id": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == "0x1234567890abcdef"  # Mocked value from WebXOSWallet

@pytest.mark.asyncio
async def test_ping_request(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.ping",
            "params": {"user_id": "test_user"},
            "id": 4
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["status"] == "pong"

@pytest.mark.asyncio
async def test_set_level_request(client):
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "mcp.setLevel",
            "params": {"user_id": "test_user", "level": "DEBUG"},
            "id": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["status"] == "Logging level set to DEBUG"
