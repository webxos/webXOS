import pytest
from fastapi.testclient import TestClient
from vial2.mcp.api import agent_endpoint

client = TestClient(agent_endpoint.app)

def test_agent_endpoint_get():
    response = client.get("/mcp/api/agent")
    assert response.status_code == 200
    assert response.json() == {"message": "Agent endpoint active"}

def test_agent_endpoint_post():
    response = client.post("/mcp/api/agent", json={"action": "start"})
    assert response.status_code == 200
    assert response.json() == {"status": "started"}