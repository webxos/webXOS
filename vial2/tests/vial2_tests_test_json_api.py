import pytest
from fastapi.testclient import TestClient
from vial2.mcp.api import json_handler

client = TestClient(json_handler.app)

def test_json_call():
    response = client.post("/mcp/tools/call", json={"tool": "test"})
    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"