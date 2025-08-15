# main/server/mcp/tests/test_webxos_balance.py
import pytest
from ..utils.webxos_balance import WebXOSBalance
from ..utils.mcp_error_handler import MCPError
import asyncio

@pytest.fixture
async def balance_manager():
    manager = WebXOSBalance()
    yield manager
    await manager.cache.close()

@pytest.mark.asyncio
async def test_update_balance(balance_manager, mocker):
    mocker.patch.object(balance_manager.cache, "set_cache", return_value=None)
    initial_balance = await balance_manager.get_balance("test_user")
    updated_balance = await balance_manager.update_balance("test_user")
    assert updated_balance != initial_balance
    assert isinstance(updated_balance, float)

@pytest.mark.asyncio
async def test_get_balance(balance_manager, mocker):
    mocker.patch.object(balance_manager.cache, "get_cache", return_value={"balance": 1000.0})
    balance = await balance_manager.get_balance("test_user")
    assert balance == 1000.0

@pytest.mark.asyncio
async def test_balance_error_handling(balance_manager, mocker):
    mocker.patch.object(balance_manager.cache, "get_cache", side_effect=Exception("Cache error"))
    with pytest.raises(MCPError) as exc_info:
        await balance_manager.get_balance("test_user")
    assert exc_info.value.code == -32603
    assert "Balance retrieval failed" in exc_info.value.message
