import pytest
import jwt
from fastapi.testclient import TestClient
from vial.auth_manager import AuthManager
import os
from unittest.mock import patch

@pytest.fixture
def auth_manager():
    return AuthManager()

@pytest.fixture
def client():
    from vial.unified_server import app
    return TestClient(app)

def test_create_token(auth_manager):
    token = auth_manager.create_token("user123", ["read:data", "read:llm"])
    decoded = jwt.decode(token, os.getenv("JWT_SECRET", "VIAL_MCP_SECRET_2025"), algorithms=["HS256"])
    assert decoded["user_id"] == "user123"
    assert decoded["roles"] == ["read:data", "read:llm"]

def test_verify_token_success(auth_manager):
    token = auth_manager.create_token("user123", ["read:data", "read:llm"])
    payload = auth_manager.verify_token(token, ["read:data"])
    assert payload["user_id"] == "user123"

def test_verify_token_insufficient_permissions(auth_manager):
    token = auth_manager.create_token("user123", ["read:data"])
    with pytest.raises(Exception) as exc:
        auth_manager.verify_token(token, ["write:git"])
    assert "Insufficient permissions" in str(exc.value)

def test_verify_token_invalid(auth_manager):
    with pytest.raises(Exception) as exc:
        auth_manager.verify_token("invalid_token", ["read:data"])
    assert "Invalid token" in str(exc.value)

@patch("requests.post")
@patch("requests.get")
def test_verify_oauth_token_success(mock_get, mock_post, auth_manager):
    mock_post.return_value.json.return_value = {"access_token": "mock_token"}
    mock_post.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"login": "user123"}
    mock_get.return_value.status_code = 200
    result = auth_manager.verify_oauth_token("mock_code")
    assert result["user_id"] == "user123"
    assert "read:data" in result["roles"]

@patch("requests.post")
def test_verify_oauth_token_failure(mock_post, auth_manager):
    mock_post.return_value.json.return_value = {}
    mock_post.return_value.status_code = 401
    with pytest.raises(Exception) as exc:
        auth_manager.verify_oauth_token("invalid_code")
    assert "OAuth token retrieval failed" in str(exc.value)

def test_generate_api_key(auth_manager, client):
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_connect.return_value.commit.return_value = None
        api_key = auth_manager.generate_api_key("user123")
        assert isinstance(api_key, str)
        mock_cursor.execute.assert_called_once()
