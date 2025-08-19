import pytest
import requests
import logging
import os

logger = logging.getLogger(__name__)

@pytest.mark.parametrize("endpoint", ["/mcp/api/health", "/mcp/api/auth"])
def test_deployment(endpoint):
    try:
        response = requests.get(f"http://localhost:8000{endpoint}")
        assert response.status_code == 200
        logger.info(f"Deployment test passed for {endpoint}")
    except Exception as e:
        logger.error(f"Deployment test failed for {endpoint}: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #deployment #neon_mcp
