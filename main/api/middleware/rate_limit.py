from fastapi import Request, HTTPException
from ...config.redis_config import get_redis
from ...utils.logging import log_error
import time

RATE_LIMIT = 1000  # Requests per minute
RATE_LIMIT_WINDOW = 60  # Seconds

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware using Redis."""
    redis = await get_redis()
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    
    try:
        current_count = await redis.get(key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= RATE_LIMIT:
            log_error(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        await redis.incr(key)
        if current_count == 0:
            await redis.expire(key, RATE_LIMIT_WINDOW)
        
        response = await call_next(request)
        return response
    except Exception as e:
        log_error(f"Rate limit middleware error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
