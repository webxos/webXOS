# main/server/mcp/utils/test_webxos_balance.py
import pytest
from ..utils.webxos_balance import WebXOSBalance
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def balance():
    return WebXOSBalance()

@pytest.mark.asyncio
async def test_get_balance(balance):
    result = await balance.get_balance("test_user")
    assert result == 500.0

@pytest.mark.asyncio
async def test_get_balance_invalid(balance):
    result = await balance.get_balance("unknown_user")
    assert result == 0.0
