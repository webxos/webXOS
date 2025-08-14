import pytest
from fastapi.testclient import TestClient
from main.server.mcp.security_manager import SecurityManager
from main.server.unified_server import app
import os
from datetime import timedelta

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def security_manager():
    """Create a SecurityManager instance."""
    return SecurityManager()

def test_create_token(security_manager, mocker):
    """Test JWT token creation."""
    mocker.patch.dict(os.environ, {"JWT_SECRET": "test_secret"})
    token = security_manager.create_token("wallet_123")
    assert isinstance(token, str)
    assert len(token.split(".")) == 3  # JWT format: header.payload.signature

def test_validate_token_success(security_manager, mocker):
    """Test successful JWT token validation."""
    mocker.patch.dict(os.environ, {"JWT_SECRET": "test_secret"})
    token = security_manager.create_token("wallet_123", expires_delta=timedelta(seconds=3600))
    payload = security_manager.validate_token(token)
    assert payload["wallet_id"] == "wallet_123"
    assert "exp" in payload

def test_validate_token_expired(security_manager, mocker):
    """Test validation of expired JWT token."""
    mocker.patch.dict(os.environ, {"JWT_SECRET": "test_secret"})
    token = security_manager.create_token("wallet_123", expires_delta=timedelta(seconds=-1))
    with pytest.raises(HTTPException) as exc:
        security_manager.validate_token(token)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Token expired"

def test_validate_token_invalid(security_manager):
    """Test validation of invalid JWT token."""
    with pytest.raises(HTTPException) as exc:
        security_manager.validate_token("invalid.token.string")
    assert exc.value.status_code == 401
    assert "Invalid token" in exc.value.detail
