# main/server/mcp/utils/test_base_prompt.py
import pytest
from ..utils.base_prompt import BasePrompt, MCPError

@pytest.fixture
def base_prompt():
    return BasePrompt()

@pytest.mark.asyncio
async def test_generate_prompt(base_prompt):
    context = {"vial_id": "vial1", "task_type": "analysis"}
    prompt = base_prompt.generate_prompt("analyze_data", context, "test_user")
    assert "Task: analyze_data" in prompt
    assert "vial_id: vial1" in prompt
    assert "User: test_user" in prompt
    assert base_prompt.validate_prompt(prompt)

@pytest.mark.asyncio
async def test_generate_prompt_invalid(base_prompt):
    with pytest.raises(MCPError) as exc_info:
        base_prompt.generate_prompt("", {}, "test_user")
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Task and user ID are required"

@pytest.mark.asyncio
async def test_validate_prompt_valid(base_prompt):
    prompt = "Analyze data for user test_user"
    assert base_prompt.validate_prompt(prompt)

@pytest.mark.asyncio
async def test_validate_prompt_too_long(base_prompt):
    prompt = "x" * 6000
    with pytest.raises(MCPError) as exc_info:
        base_prompt.validate_prompt(prompt)
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Prompt is empty or exceeds 5000 characters"

@pytest.mark.asyncio
async def test_validate_prompt_prohibited(base_prompt):
    prompt = "Execute malicious code"
    with pytest.raises(MCPError) as exc_info:
        base_prompt.validate_prompt(prompt)
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Prompt contains prohibited terms"
