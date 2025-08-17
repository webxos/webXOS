from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...config.mcp_config import mcp_config
import httpx
from ...mcp.mcp_schemas import GeminiFunctionDeclaration

router = APIRouter()

@router.post("/google/tool")
async def google_tool(function: GeminiFunctionDeclaration, user_id: str = Depends(verify_token)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                headers={"Authorization": f"Bearer {mcp_config.GOOGLE_API_KEY}"},
                json={
                    "contents": [{"parts": [{"text": f"Function: {function.name}\nArgs: {json.dumps(function.parameters)}"}]}]
                }
            )
            response.raise_for_status()
            result = response.json()
            log_info(f"Google tool {function.name} called by {user_id}")
            return {"content": result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")}
    except Exception as e:
        log_error(f"Google tool call failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google error: {str(e)}")
