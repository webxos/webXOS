from fastmcp.client import Client
import asyncio
import logging

logger = logging.getLogger(__name__)

async def run_client(mcp_url: str = "http://localhost:5000"):
    """Run MCP client to interact with Vial MCP server."""
    try:
        async with Client(mcp_url) as client:
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            result = await client.call_tool("train_vials", {"networkId": "test", "file": "test.txt"})
            logger.info(f"Train result: {result}")
            return result
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-11T05:48:00Z]** Client error: {str(e)}\n")
        raise

if __name__ == "__main__":
    asyncio.run(run_client())