import pytest
from ..langchain.mcp_chain import mcp_chain
import logging
import os

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_langchain_integration():
    try:
        os.environ["GROK_API_KEY"] = "test_key"
        await mcp_chain._call({"query": "Test query"})
        logger.info("LangChain integration test passed")
    except Exception as e:
        logger.error(f"LangChain integration test failed: {str(e)}")
        raise
    finally:
        if "GROK_API_KEY" in os.environ:
            del os.environ["GROK_API_KEY"]

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #integration #neon_mcp
