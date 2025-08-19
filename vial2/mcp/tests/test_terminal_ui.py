import pytest
from mcp.ui.terminal_interface import terminal_interface
import logging

logger = logging.getLogger(__name__)

def test_terminal_render():
    try:
        output = terminal_interface.render("test_command")
        assert "Executing" in output
        logger.info("Terminal UI test passed")
    except Exception as e:
        logger.error(f"Terminal UI test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #terminal #ui #neon_mcp
