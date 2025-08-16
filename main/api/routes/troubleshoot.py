from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis
from ...security.authentication import verify_token
import sqlite3
import redis.asyncio as redis

router = APIRouter(prefix="/v1/troubleshoot", tags=["Troubleshoot"])

@router.get("/")
async def troubleshoot_system(user_id: str = Depends(verify_token), redis=Depends(get_redis)):
    """Run system diagnostics."""
    try:
        diagnostics = {"status": "healthy", "components": {}}
        
        # Check Redis
        try:
            await redis.ping()
            diagnostics["components"]["redis"] = "connected"
        except redis.ConnectionError as e:
            diagnostics["components"]["redis"] = f"failed: {str(e)}"
            log_error(f"Redis connection check failed: {str(e)}")
        
        # Check SQLite
        try:
            conn = sqlite3.connect("main/errors.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM error_logs")
            diagnostics["components"]["sqlite"] = f"connected, {cursor.fetchone()[0]} logs"
            conn.close()
        except sqlite3.Error as e:
            diagnostics["components"]["sqlite"] = f"failed: {str(e)}"
            log_error(f"SQLite connection check failed: {str(e)}")
        
        log_info(f"Diagnostics run by user {user_id}: {diagnostics}")
        return diagnostics
    except Exception as e:
        log_error(f"Troubleshoot failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
