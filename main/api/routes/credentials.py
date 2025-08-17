from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...config.redis_config import get_redis
from ..mcp.handlers.tools import ToolHandler

router = APIRouter()
tool_handler = ToolHandler()

@router.get("/generate-credentials")
async def generate_credentials(user_id: str = Depends(verify_token), redis=Depends(get_redis)):
    try:
        credentials = await tool_handler.handle_generate_credentials(user_id, redis)
        log_info(f"Credentials generated for {user_id}")
        return {"status": "success", "credentials": credentials}
    except Exception as e:
        log_error(f"Credentials generation failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credentials error: {str(e)}")
