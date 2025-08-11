import pytest
from langchain_agent import create_langchain_agent

@pytest.mark.asyncio
async def test_langchain_agent():
    agent = create_langchain_agent()
    response = await agent.arun("test input")
    assert "Simulated NanoGPT response to: test input" in response