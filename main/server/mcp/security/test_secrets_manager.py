# main/server/mcp/security/test_secrets_manager.py
import pytest
from pymongo import MongoClient
from ..security.secrets_manager import SecretsManager, MCPError

@pytest.fixture
async def secrets_manager():
    manager = SecretsManager()
    yield manager
    manager.collection.delete_many({})
    await manager.redis_client.flushdb()
    manager.close()

@pytest.mark.asyncio
async def test_store_secret(secrets_manager):
    secret_id = await secrets_manager.store_secret("test_user", "api_key", "secret_value")
    assert isinstance(secret_id, str)
    assert len(secret_id) == 32

@pytest.mark.asyncio
async def test_store_secret_invalid(secrets_manager):
    with pytest.raises(MCPError) as exc_info:
        await secrets_manager.store_secret("test_user", "", "secret_value")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Secret name and value are required"

@pytest.mark.asyncio
async def test_retrieve_secret(secrets_manager):
    secret_id = await secrets_manager.store_secret("test_user", "api_key", "secret_value")
    retrieved = await secrets_manager.retrieve_secret("test_user", secret_id)
    assert retrieved == "secret_value"

@pytest.mark.asyncio
async def test_retrieve_secret_not_found(secrets_manager):
    with pytest.raises(MCPError) as exc_info:
        await secrets_manager.retrieve_secret("test_user", "invalid_id")
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Secret not found or access denied"

@pytest.mark.asyncio
async def test_delete_secret(secrets_manager):
    secret_id = await secrets_manager.store_secret("test_user", "api_key", "secret_value")
    await secrets_manager.delete_secret("test_user", secret_id)
    with pytest.raises(MCPError) as exc_info:
        await secrets_manager.retrieve_secret("test_user", secret_id)
    assert exc_info.value.code == -32003
