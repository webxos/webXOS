import pytest
from mcp.offline.offline_handler import offline_handler
import logging
import json

logger = logging.getLogger(__name__)

def test_offline_queue():
    try:
        offline_handler.queue_offline_request({"type": "test"})
        with open("offline_queue.json", "r") as f:
            queue = json.load(f)
        assert len(queue) > 0
        logger.info("Offline mode test passed")
    except Exception as e:
        logger.error(f"Offline mode test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #offline #mode #neon_mcp
