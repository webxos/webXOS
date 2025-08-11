from fastmcp import mcp
import logging

logger = logging.getLogger(__name__)

@mcp.tool
def sample_tool(data: str) -> str:
    """Sample MCP tool for processing input data."""
    try:
        result = f"Processed: {data}"
        logger.info(f"Sample tool processed: {data}")
        return result
    except Exception as e:
        logger.error(f"Sample tool error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-11T05:48:00Z]** Sample tool error: {str(e)}\n")
        raise