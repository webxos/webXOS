import pytest
from fastapi.testclient import TestClient
from main.server.mcp.api_config import APIConfig
from main.server.unified_server import app
import os

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def api_config():
    """Create an APIConfig instance."""
    return APIConfig()

def test_get_config(api_config, mocker):
    """Test retrieving API configuration."""
    mocker.patch.dict(os.environ, {
        "API_HOST": "localhost",
        "PYTHON_DOCKER_PORT": "8000",
        "LOG_LEVEL": "debug",
        "ALLOWED_ORIGINS": "https://webxos.netlify.app",
        "MAX_REQUEST_SIZE": "2000000",
        "RATE_LIMIT_REQUESTS": "200",
        "RATE_LIMIT_WINDOW": "120"
    })
    config = api_config.get_config()
    assert config["host"] == "localhost"
    assert config["port"] == 8000
    assert config["log_level"] == "debug"
    assert config["allowed_origins"] == ["https://webxos.netlify.app"]
    assert config["max_request_size"] == 2000000
    assert config["rate_limit_requests"] == 200
    assert config["rate_limit_window"] == 120

def test_update_config_success(api_config):
    """Test updating a valid configuration key."""
    api_config.update_config("log_level", "error")
    assert api_config.get_config()["log_level"] == "error"

def test_update_config_invalid_key(api_config):
    """Test updating an invalid configuration key."""
    with pytest.raises(HTTPException) as exc:
        api_config.update_config("invalid_key", "value")
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid config key: invalid_key"
