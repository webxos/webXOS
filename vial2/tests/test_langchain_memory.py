import pytest
from ..langchain.memory_manager import memory_manager
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_memory_manager():
    try:
        memory = memory_manager.get_memory("vial1")
        memory.save_context({"input": "Test"}, {"output": "Response"})
        assert memory.load_memory_variables({})["history"] == "Human: Test\nAI: Response"
        logger.info("LangChain memory test passed")
    except Exception as e:
        logger.error(f"LangChain memory test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #memory #neon_mcp
