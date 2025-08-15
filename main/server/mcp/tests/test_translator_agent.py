# main/server/mcp/tests/test_translator_agent.py
import pytest
from ..agents.translator_agent import TranslatorAgent  # Assume implementation exists
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def translator_agent():
    return TranslatorAgent()

@pytest.mark.asyncio
async def test_translate(translator_agent, mocker):
    mocker.patch.object(translator_agent, "translate", return_value="Translated text")
    result = await translator_agent.translate("Hello", "es")
    assert result == "Translated text"

@pytest.mark.asyncio
async def test_translate_error(translator_agent, mocker):
    mocker.patch.object(translator_agent, "translate", side_effect=MCPError(code=-32603, message="Translation failed"))
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate("Hello", "invalid_lang")
    assert exc_info.value.code == -32603