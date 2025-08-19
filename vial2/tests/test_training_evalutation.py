import pytest
from ..langchain.training_evaluator import training_evaluator
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_training_evaluation():
    try:
        score = await training_evaluator.evaluate_training("vial1", ["git status", "git log"])
        assert score is not None
        logger.info("Training evaluation test passed")
    except Exception as e:
        logger.error(f"Training evaluation test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #training #evaluation #neon_mcp
