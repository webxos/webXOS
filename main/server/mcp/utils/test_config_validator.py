# main/server/mcp/utils/test_config_validator.py
import pytest
from ..utils.config_validator import ConfigValidator, MCPError
import os

@pytest.fixture
def config_validator():
    return ConfigValidator()

@pytest.mark.asyncio
async def test_validate_config_valid(config_validator):
    config = {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": 8080,
        "MONGODB_URI": "mongodb://localhost:27017",
        "REDIS_URI": "redis://localhost:6379",
        "SECRET_KEY": "secure_random_key_32_chars_long",
        "JWT_ALGORITHM": "HS256",
        "ALLOWED_ORIGINS": ["http://localhost:3000"]
    }
    config_validator.validate_config(config)  # Should not raise

@pytest.mark.asyncio
async def test_validate_config_missing_key(config_validator):
    config = {
        "MCP_SERVER_HOST": "0.0.0.0",
        "MCP_SERVER_PORT": 8080,
        # Missing MONGODB_URI
        "REDIS_URI": "redis://localhost:6379",
        "SECRET_KEY": "secure_random_key_32_chars_long",
        "JWT_ALGORITHM": "HS256",
        "ALLOWED_ORIGINS": "http://localhost:3000"
    }
    with pytest.raises(MCPError) as exc_info:
        config_validator.validate_config(config)
    assert exc_info.value.code == -32602
    assert "Missing or empty configuration: MONGODB_URI" in exc_info.value.message

@pytest.mark.asyncio
async def test_validate_token_valid(config_validator, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert config_validator.validate_token("valid_token_32_chars_abcdefghijklmnop", "repo:public")

@pytest.mark.asyncio
async def test_validate_token_invalid_scope(config_validator, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(MCPError) as exc_info:
        config_validator.validate_token("valid_token_32_chars_abcdefghijklmnop", "repo:all")
    assert exc_info.value.code == -32602
    assert "Broad scope 'repo:all' not allowed in production" in exc_info.value.message

@pytest.mark.asyncio
async def test_sanitize_config(config_validator):
    config = {
        "MCP_SERVER_HOST": "0.0.0.0",
        "SECRET_KEY": "secure_random_key_32_chars_long",
        "TRANSLATION_API_KEY": "api_key_123"
    }
    sanitized = config_validator.sanitize_config(config)
    assert sanitized["MCP_SERVER_HOST"] == "0.0.0.0"
    assert sanitized["SECRET_KEY"] == "****"
    assert sanitized["TRANSLATION_API_KEY"] == "****"
