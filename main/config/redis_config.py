import redis.asyncio as redis
from ..config.settings import settings

async def get_redis():
    """Initialize Redis connection with async support."""
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True
        )
        await client.ping()
        return client
    except redis.ConnectionError as e:
        from ..utils.logging import log_error
        log_error(f"Redis connection failed: {str(e)}")
        raise

async def close_redis():
    """Close Redis connection."""
    client = await get_redis()
    await client.aclose()
