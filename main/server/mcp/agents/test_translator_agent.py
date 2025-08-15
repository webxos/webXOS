# main/server/mcp/agents/test_translator_agent.py
import pytest
from ..agents.translator_agent import TranslatorAgent, MCPError
import aiohttp
import json

@pytest.fixture
async def translator_agent():
    agent = TranslatorAgent()
    yield agent
    agent.close()

@pytest.mark.asyncio
async def test_translate_config(translator_agent, mocker):
    mocker.patch("aiohttp.ClientSession.post", return_value=mocker.AsyncMock(
        status=200,
        json=mocker.AsyncMock(return_value={"translated_text": "Hola"})
    ))
    config = {"greeting": "Hello", "details": {"message": "World"}}
    result = await translator_agent.translate_config(config, "es")
    assert result["greeting"] == "Hola"
    assert result["details"]["message"] == "Hola"

@pytest.mark.asyncio
async def test_translate_config_invalid(translator_agent):
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate_config({}, "es")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Config and target language are required"

@pytest.mark.asyncio
async def test_translate_prompt(translator_agent, mocker):
    mocker.patch("aiohttp.ClientSession.post", return_value=mocker.AsyncMock(
        status=200,
        json=mocker.AsyncMock(return_value={"translated_text": "Bonjour"})
    ))
    result = await translator_agent.translate_prompt("Hello", "fr", "test_user")
    assert result == "Bonjour"

@pytest.mark.asyncio
async def test_translate_prompt_unsupported_language(translator_agent):
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate_prompt("Hello", "xx", "test_user")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Unsupported language: xx"

@pytest.mark.asyncio
async def test_translate_api_error(translator_agent, mocker):
    mocker.patch("aiohttp.ClientSession.post", return_value=mocker.AsyncMock(status=500))
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate_prompt("Hello", "es", "test_user")
    assert exc_info.value.code == -32603
    assert "Translation API error" in exc_info.value.message
