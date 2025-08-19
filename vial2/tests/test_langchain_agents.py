import pytest
from ..langchain.agent_executor import agent_executor
from ..langchain.tool_manager import tool_manager
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_agent_execution():
    try:
        def mock_tool(input):
            return f"Mock response for {input}"
        tool_manager.register_tool("grok_mock", mock_tool)
        result = await agent_executor.execute("Test query")
        assert "Mock response for Test query" in result
        logger.info("LangChain agent test passed")
    except Exception as e:
        logger.error(f"LangChain agent test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #agents #neon_mcp
