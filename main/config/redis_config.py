import aioredis
from ...config.mcp_config import mcp_config
from ...utils.logging import log_error, log_info

async def get_redis():
    try:
        redis = await aioredis.from_url(
            mcp_config.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=100
        )
        log_info("Redis connection established")
        return redis
    except Exception as e:
        log_error(f"Redis connection failed: {str(e)}")
        raise
