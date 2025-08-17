from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...config.mcp_config import mcp_config
import httpx
from ...mcp.mcp_schemas import OpenAIToolCall

router = APIRouter()

@router.post("/openai/tool")
async def openai_tool(tool_call: OpenAIToolCall, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {mcp_config.OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": f"Tool: {tool_call.function['name']}\nArgs: {json.dumps(tool_call.function['arguments'])}"}],
                    "tools": [{"type": "function", "function": tool_call.function}]
                }
            )
            response.raise_for_status()
            result = response.json()
            log_info(f"OpenAI tool {tool_call.function['name']} called by {user_id}")
            return {"content": result.get("choices", [{}])[0].get("message", {}).get("content", "")}
    except Exception as e:
        log_error(f"OpenAI tool call failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
