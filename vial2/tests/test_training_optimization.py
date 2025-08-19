import pytest
from ..langchain.training_optimizer import training_optimizer
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_training_optimization():
    try:
        score = await training_optimizer.optimize_training("vial1", ["git status", "git log"])
        assert score is not None
        logger.info("Training optimization test passed")
    except Exception as e:
        logger.error(f"Training optimization test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #training #optimization #neon_mcp
