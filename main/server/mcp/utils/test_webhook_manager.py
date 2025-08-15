# main/server/mcp/utils/test_webhook_manager.py
import pytest
from ..utils.webhook_manager import WebhookManager
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def webhook_manager():
    return WebhookManager()

@pytest.mark.asyncio
async def test_send_webhook(webhook_manager, mocker):
    mocker.patch.object(webhook_manager, "send_webhook", return_value=True)
    result = await webhook_manager.send_webhook("http://example.com", {"test": "data"})
    assert result is True

@pytest.mark.asyncio
async def test_send_webhook_error(webhook_manager, mocker):
    mocker.patch.object(webhook_manager, "send_webhook", side_effect=MCPError(code=-32603, message="Webhook failed"))
    with pytest.raises(MCPError) as exc_info:
        await webhook_manager.send_webhook("http://example.com", {})
    assert exc_info.value.code == -32603
