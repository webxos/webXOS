import pytest
from ..api.json_handler import json_handler
import logging
import json

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_json_parsing():
    try:
        valid_json = '{"jsonrpc": "2.0", "method": "agent", "params": {"agent_type": "grok"}, "id": "1"}'
        result = json_handler.parse_json(valid_json)
        assert result.method == "agent"
        logger.info("JSON API parsing test passed")
    except Exception as e:
        logger.error(f"JSON API parsing test failed: {str(e)}")
        raise

    try:
        invalid_json = '{"jsonrpc": "1.0", "method": "invalid", "params": {}, "id": "2"}'
        with pytest.raises(Exception):
            json_handler.parse_json(invalid_json)
        logger.info("JSON API invalid parsing test passed")
    except Exception as e:
        logger.error(f"JSON API invalid parsing test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #api #json #neon_mcp
