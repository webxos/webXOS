# main/server/mcp/utils/test_api_docs.py
import pytest
from ..utils.api_docs import APIDocs, MCPError
import json

@pytest.fixture
def api_docs():
    return APIDocs()

@pytest.mark.asyncio
async def test_generate_docs(api_docs, mocker):
    mocker.patch.object(api_docs.api_config, "load_config", return_value={
        "endpoints": {
            "mcp.createSession": {"enabled": True, "auth_required": True},
            "mcp.initiateMFA": {"enabled": True, "auth_required": False}
        }
    })
    docs = api_docs.generate_docs()
    assert "# Vial MCP Controller API Documentation" in docs
    assert "### mcp.createSession" in docs
    assert "Auth Required: True" in docs
    assert "user_id (string, required)" in docs
    assert "```json" in docs

@pytest.mark.asyncio
async def test_generate_docs_no_endpoints(api_docs, mocker):
    mocker.patch.object(api_docs.api_config, "load_config", return_value={"endpoints": {}})
    docs = api_docs.generate_docs()
    assert "# Vial MCP Controller API Documentation" in docs
    assert "### mcp.createSession" not in docs

@pytest.mark.asyncio
async def test_generate_docs_error(api_docs, mocker):
    mocker.patch.object(api_docs.api_config, "load_config", side_effect=Exception("Config error"))
    with pytest.raises(MCPError) as exc_info:
        api_docs.generate_docs()
    assert exc_info.value.code == -32603
    assert "Failed to generate API docs" in exc_info.value.message
