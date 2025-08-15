# main/server/mcp/tests/test_error_handler.py
import pytest
from ..utils.error_handler import handle_error, MCPError

def test_handle_error():
    error = Exception("Test error")
    result = handle_error(error)
    assert "error" in result
    assert result["error"]["code"] == -32603
    assert "Test error" in result["error"]["message"]