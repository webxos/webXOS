from fastapi import APIRouter, Depends, HTTPException
from ..utils.logging import log_error, log_info
from ..utils.authentication import verify_token
from ..config.mcp_config import mcp_config
import httpx

router = APIRouter()

@router.post("/xai/tool")
async def xai_tool(tool_call: dict, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.x.ai/v1/grok",
                headers={"Authorization": f"Bearer {mcp_config.XAI_API_KEY}"},
                json={
                    "model": "grok-3",
                    "messages": [{"role": "user", "content": f"Tool: {tool_call.get('name')}\nArgs: {json.dumps(tool_call.get('arguments', {}))}"}]
                }
            )
            response.raise_for_status()
            result = response.json()
            log_info(f"xAI tool {tool_call.get('name')} called by {user_id}")
            return {"content": result.get("choices", [{}])[0].get("message", {}).get("content", "")}
    except Exception as e:
        log_error(f"xAI tool call failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"xAI error: {str(e)}")
