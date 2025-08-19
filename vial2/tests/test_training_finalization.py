import pytest
from ..langchain.training_finalizer import training_finalizer
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_training_finalization():
    try:
        result = await training_finalizer.finalize_training("vial1", ["git status", "git log"])
        assert result["status"] == "completed"
        logger.info("Training finalization test passed")
    except Exception as e:
        logger.error(f"Training finalization test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #training #finalization #neon_mcp
