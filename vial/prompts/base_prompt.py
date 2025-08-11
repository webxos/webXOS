from fastmcp import mcp
import logging

logger = logging.getLogger(__name__)

@mcp.prompt
def base_prompt(input: str) -> str:
    """Sample prompt template for LangChain."""
    try:
        result = f"Analyze this: {input}"
        logger.info(f"Base prompt processed: {input}")
        return result
    except Exception as e:
        logger.error(f"Base prompt error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-11T05:48:00Z]** Base prompt error: {str(e)}\n")
        raise