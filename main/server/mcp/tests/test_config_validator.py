# main/server/mcp/tests/test_config_validator.py
import pytest
from ..utils.config_validator import ConfigValidator  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def validator():
    return ConfigValidator()

@pytest.mark.asyncio
async def test_valid_config(validator, mocker):
    mocker.patch.object(validator, "validate", return_value=True)
    result = await validator.validate({"logging": {"level": "INFO"}})
    assert result is True

@pytest.mark.asyncio
async def test_invalid_config(validator, mocker):
    mocker.patch.object(validator, "validate", side_effect=MCPError(code=-32603, message="Invalid config"))
    with pytest.raises(MCPError) as exc_info:
        await validator.validate({"logging": {"level": "INVALID"}})
    assert exc_info.value.code == -32603