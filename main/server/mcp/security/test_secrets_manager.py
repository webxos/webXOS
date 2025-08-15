# main/server/mcp/security/test_secrets_manager.py
import pytest
from ..security.secrets_manager import SecretsManager

@pytest.fixture
def secrets_manager():
    return SecretsManager()

@pytest.mark.asyncio
async def test_get_secret(secrets_manager):
    secret = await secrets_manager.get_secret("SECRET_KEY")
    assert secret == "default_secret_key"
