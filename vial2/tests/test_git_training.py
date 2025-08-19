import pytest
from ..langchain.git_training import git_training
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_git_training():
    try:
        await git_training.train_with_git("vial1", ["git commit", "git push"])
        logger.info("Git training test passed")
    except Exception as e:
        logger.error(f"Git training test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #git #training #neon_mcp
