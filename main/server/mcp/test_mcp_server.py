# main/server/mcp/test_mcp_server.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from ..mcp_server import app, VialMCPServer

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_jsonrpc_version(client):
    response = await client.post("/mcp", json={"jsonrpc": "1.0", "method": "mcp.getCapabilities", "id": 1})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32600
    assert response.json()["error"]["message"] == "Invalid JSON-RPC version"

@pytest.mark.asyncio
async def test_get_capabilities(client):
    response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "mcp.getCapabilities", "id": 1})
    assert response.status_code == 200
    result = response.json()["result"]
    assert "resources" in result
    assert "tools" in result
    assert result["version"] == "1.0.0"
    assert result["protocol"] == "mcp/1.0"

@pytest.mark.asyncio
async def test_list_resources(client):
    response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "mcp.listResources", "id": 2})
    assert response.status_code == 200
    resources = response.json()["result"]
    assert len(resources) > 0
    assert any(r["uri"].startswith("vial://notes/") for r in resources)

@pytest.mark.asyncio
async def test_call_tool(client):
    response = await client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "mcp.callTool",
        "params": {"tool_name": "create_note", "title": "Test Note", "content": "Test Content", "tags": ["test"]},
        "id": 3
    })
    assert response.status_code == 200
    assert "result" in response.json()

@pytest.mark.asyncio
async def test_invalid_tool(client):
    response = await client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "mcp.callTool",
        "params": {"tool_name": "invalid_tool"},
        "id": 4
    })
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32601
    assert response.json()["error"]["message"] == "Tool invalid_tool not found"

@pytest.mark.asyncio
async def test_invalid_method(client):
    response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "invalid.method", "id": 5})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32601
    assert response.json()["error"]["message"] == "Method not found"
