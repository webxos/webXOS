import pytest
from fastapi.testclient import TestClient
from vial.unified_server import app
from unittest.mock import patch
import jwt
import os

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def token():
    return jwt.encode(
        {"user_id": "user123", "roles": ["read:data", "read:llm", "write:git"]},
        os.getenv("JWT_SECRET", "VIAL_MCP_SECRET_2025"),
        algorithm="HS256"
    )

def test_retrieve_endpoint(client, token):
    response = client.post(
        "/v1/api/retrieve",
        json={"user_id": "user123", "query": "test", "source": "postgres", "format": "json", "wallet": {}},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "success"

def test_retrieve_unauthorized(client):
    response = client.post(
        "/v1/api/retrieve",
        json={"user_id": "user123", "query": "test", "source": "postgres", "format": "json", "wallet": {}}
    )
    assert response.status_code == 401
    assert "Unauthorized" in response.json().get("error")

def test_llm_endpoint(client, token):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "mocked"}
        mock_post.return_value.status_code = 200
        response = client.post(
            "/v1/api/llm",
            json={"user_id": "user123", "prompt": "test", "model": "llama3.3", "format": "json", "wallet": {}},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json().get("status") == "success"

def test_git_endpoint(client, token):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "git status output"
        mock_run.return_value.returncode = 0
        response = client.post(
            "/v1/api/git",
            json={"user_id": "user123", "command": "git status", "repo_url": "https://github.com/webxos/webxos.git"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json().get("status") == "success"

def test_rate_limit(client, token):
    for _ in range(10):
        response = client.post(
            "/v1/api/retrieve",
            json={"user_id": "user123", "query": "test", "source": "postgres", "format": "json", "wallet": {}},
            headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json().get("error")
