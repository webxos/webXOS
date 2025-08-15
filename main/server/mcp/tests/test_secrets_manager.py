# main/server/mcp/tests/test_secrets_manager.py
import pytest
from ..security.secrets_manager import SecretsManager  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def secrets_manager():
    return SecretsManager()

@pytest.mark.asyncio
async def test_get_secret(secrets_manager, mocker):
    mocker.patch.object(secrets_manager, "get_secret", return_value="test_secret")
    secret = await secrets_manager.get_secret("test_key")
    assert secret == "test_secret"

@pytest.mark.asyncio
async def test_missing_secret(secrets_manager, mocker):
    mocker.patch.object(secrets_manager, "get_secret", side_effect=MCPError(code=-32004, message="Secret not found"))
    with pytest.raises(MCPError) as exc_info:
        await secrets_manager.get_secret("nonexistent")
    assert exc_info.value.code == -32004