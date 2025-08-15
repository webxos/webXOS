# main/server/mcp/utils/test_base_prompt.py
import pytest
from ..utils.base_prompt import BasePrompt, MCPError
from ..utils.performance_metrics import PerformanceMetrics

@pytest.fixture
async def base_prompt():
    prompt = BasePrompt()
    yield prompt
    await prompt.close()

@pytest.mark.asyncio
async def test_generate_prompt(base_prompt, mocker):
    mocker.patch("aiohttp.ClientSession.post", return_value=mocker.AsyncMock(
        status=200,
        json=mocker.AsyncMock(return_value={"translated_text": "Hola"})
    ))
    request = {"method": "test_method", "params": {"data": "value"}}
    prompt = await base_prompt.generate_prompt(request, "test_user", "es")
    assert "system" in prompt
    assert "Hola" in prompt["user"]
    assert prompt["constraints"]["language"] == "es"
    assert base_prompt.metrics.requests_total.labels(endpoint="generate_prompt")._value.get() == 1

@pytest.mark.asyncio
async def test_generate_prompt_default_language(base_prompt):
    request = {"method": "test_method", "params": {"data": "value"}}
    prompt = await base_prompt.generate_prompt(request, "test_user")
    assert "system" in prompt
    assert json.dumps(request) in prompt["user"]
    assert prompt["constraints"]["language"] == "en"

@pytest.mark.asyncio
async def test_validate_prompt_valid(base_prompt):
    prompt = {
        "system": "System message",
        "user": "User request",
        "constraints": {"max_tokens": 1000, "response_format": "json", "language": "en"}
    }
    result = await base_prompt.validate_prompt(prompt)
    assert result is True
    assert base_prompt.metrics.requests_total.labels(endpoint="validate_prompt")._value.get() == 1

@pytest.mark.asyncio
async def test_validate_prompt_invalid(base_prompt):
    prompt = {
        "system": "System message",
        "constraints": {"max_tokens": "invalid", "response_format": "json"}
    }
    with pytest.raises(MCPError) as exc_info:
        await base_prompt.validate_prompt(prompt)
    assert exc_info.value.code == -32602
    assert "Prompt missing required keys" in exc_info.value.message
