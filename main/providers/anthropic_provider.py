from fastapi import APIRouter, Depends, HTTPException
from ..utils.logging import log_error, log_info
from ..utils.authentication import verify_token
from ..config.mcp_config import mcp_config
import httpx
from ..mcp_schemas import ClaudeToolUse, ClaudeToolResult

router = APIRouter()

@router.post("/anthropic/tool")
async def anthropic_tool(tool_use: ClaudeToolUse, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/complete",
                headers={"Authorization": f"Bearer {mcp_config.ANTHROPIC_API_KEY}"},
                json={
                    "model": "claude-3-opus-20240229",
                    "prompt": f"Tool use: {tool_use.name}\nInput: {json.dumps(tool_use.input)}",
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            result = response.json()
            log_info(f"Anthropic tool {tool_use.name} called by {user_id}")
            return ClaudeToolResult(content=result.get("completion", ""))
    except Exception as e:
        log_error(f"Anthropic tool call failed for {user_id}: {str(e)}")
        return ClaudeToolResult(content=str(e), isError=True)
