from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ...security.authentication import verify_token
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis

router = APIRouter(prefix="/v1/mcp", tags=["MCP Endpoints"])

class ConfigResponse(BaseModel):
    rate_limit: int
    enabled: bool

@router.get("/config", response_model=ConfigResponse)
async def get_mcp_config(user_id: str = Depends(verify_token), redis=Depends(get_redis)):
    """Retrieve MCP configuration."""
    try:
        cached_config = await redis.get(f"config:{user_id}")
        if cached_config:
            log_info(f"MCP config cache hit for user {user_id}")
            return ConfigResponse(**json.loads(cached_config))
        
        config = {"rate_limit": 1000, "enabled": True}
        await redis.set(f"config:{user_id}", json.dumps(config), ex=3600)
        log_info(f"MCP config fetched for user {user_id}")
        return ConfigResponse(**config)
    except Exception as e:
        log_error(f"MCP config fetch failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
