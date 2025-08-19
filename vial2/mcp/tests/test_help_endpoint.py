import pytest
import requests
import logging
import os

logger = logging.getLogger(__name__)

def test_help_endpoint():
    try:
        response = requests.get(f"http://localhost:8000/mcp/api/help", headers={"Authorization": f"Bearer {os.getenv('TEST_TOKEN')}"})
        assert response.status_code == 200
        assert "Commands" in response.json()["result"]["help"]
        logger.info("Help endpoint test passed")
    except Exception as e:
        logger.error(f"Help endpoint test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #help #neon_mcp
