import pytest
from ..langchain.training_monitor import training_monitor
import logging
import asyncio

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_training_monitoring():
    try:
        status = await training_monitor.monitor_training("vial1", ["git status", "git log"])
        assert status is not None
        logger.info("Training monitoring test passed")
    except Exception as e:
        logger.error(f"Training monitoring test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #langchain #training #monitoring #neon_mcp
