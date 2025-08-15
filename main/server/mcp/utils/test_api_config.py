# main/server/mcp/utils/test_api_config.py
import pytest
from ..utils.api_config import APIConfig, MCPError
import os
import json

@pytest.fixture
def api_config():
    return APIConfig()

@pytest.mark.asyncio
async def test_load_default_config(api_config, mocker):
    mocker.patch("os.path.exists", return_value=False)
    config = api_config.load_config()
    assert "endpoints" in config
    assert config["endpoints"]["mcp.createSession"]["enabled"] is True
    assert "cors" in config
    assert "http://localhost:3000" in config["cors"]["allowed_origins"]

@pytest.mark.asyncio
async def test_load_file_config(api_config, mocker, tmp_path):
    config_file = tmp_path / "api_config.json"
    custom_config = {
        "endpoints": {"mcp.test": {"enabled": True, "auth_required": False}},
        "cors": {"allowed_origins": ["http://example.com"]}
    }
    config_file.write_text(json.dumps(custom_config))
    mocker.patch("os.getenv", return_value=str(config_file))
    config = api_config.load_config()
    assert config["endpoints"]["mcp.test"]["enabled"] is True
    assert config["cors"]["allowed_origins"] == ["http://example.com"]

@pytest.mark.asyncio
async def test_validate_endpoint(api_config):
    assert api_config.validate_endpoint("mcp.createSession") is True
    assert api_config.validate_endpoint("mcp.invalid") is False

@pytest.mark.asyncio
async def test_get_endpoint_config(api_config):
    config = api_config.get_endpoint_config("mcp.createSession")
    assert config["enabled"] is True
    assert config["auth_required"] is True

@pytest.mark.asyncio
async def test_get_cors_config(api_config):
    cors = api_config.get_cors_config()
    assert "allowed_origins" in cors
    assert cors["allow_credentials"] is True
