from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from config.config import DatabaseConfig
from lib.security import SecurityHandler
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
import json
import redis.asyncio as redis
import os
import asyncio

logger = logging.getLogger("mcp.monitoring")
logger.setLevel(logging.INFO)

router = APIRouter()

class MonitoringHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security_handler = SecurityHandler(db)
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.websocket_limit_key_prefix = "websocket_limit:"
        self.websocket_limit = 5  # Max 5 concurrent WebSocket connections per user

    async def get_security_kpis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        try:
            time_window = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            auth_success = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["auth_success", time_window]
            )
            auth_failure = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["auth_error", time_window]
            )
            
            token_validations = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["token_verified", time_window]
            )
            
            active_sessions = await self.db.query(
                "SELECT COUNT(*) FROM sessions WHERE expires_at > $1",
                [datetime.utcnow()]
            )
            
            anomalies = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["anomaly_detected", time_window]
            )
            
            kpis = {
                "auth_success_rate": auth_success.rows[0]["count"] / (auth_success.rows[0]["count"] + auth_failure.rows[0]["count"] + 1) * 100,
                "auth_failure_count": auth_failure.rows[0]["count"],
                "token_validations": token_validations.rows[0]["count"],
                "active_sessions": active_sessions.rows[0]["count"],
                "anomalies_detected": anomalies.rows[0]["count"]
            }
            
            await self.security_handler.log_event(
                event_type="monitoring_kpis",
                user_id=None,
                details={"kpis": kpis, "time_window_hours": time_window_hours}
            )
            await self.security_handler.log_user_action(
                user_id=None,
                action="kpi_access",
                details={"time_window_hours": time_window_hours}
            )
            logger.info(f"Generated security KPIs for last {time_window_hours} hours")
            return kpis
        except Exception as e:
            logger.error(f"Error generating KPIs: {str(e)}")
            await self.security_handler.log_event(
                event_type="monitoring_error",
                user_id=None,
                details={"error": str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def enforce_websocket_limit(self, user_id: str):
        """Enforce WebSocket connection limit per user."""
        key = f"{self.websocket_limit_key_prefix}{user_id}"
        current_count = await self.redis_client.get(key)
        count = int(current_count) if current_count else 0
        if count >= self.websocket_limit:
            raise WebSocketDisconnect(code=1008, reason="WebSocket connection limit reached")
        await self.redis_client.incr(key)
        await self.redis_client.expire(key, 3600)  # Expire after 1 hour
        return key

    async def stream_kpis(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        redis_key = None
        try:
            redis_key = await self.enforce_websocket_limit(user_id)
            while True:
                kpis = await self.get_security_kpis(time_window_hours=1)
                await websocket.send_json({"type": "kpi_update", "data": kpis})
                await asyncio.sleep(60)  # Update every minute
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user_id}")
            await self.security_handler.log_event(
                event_type="websocket_disconnect",
                user_id=user_id,
                details={"reason": "client disconnect"}
            )
        except Exception as e:
            logger.error(f"Error streaming KPIs: {str(e)}")
            await self.security_handler.log_event(
                event_type="kpi_stream_error",
                user_id=user_id,
                details={"error": str(e)}
            )
        finally:
            if redis_key:
                await self.redis_client.decr(redis_key)
            await websocket.close()

@router.get("/monitoring/kpis", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def get_kpis(time_window_hours: int = 24, handler: MonitoringHandler = Depends(lambda: MonitoringHandler(DatabaseConfig()))):
    return await handler.get_security_kpis(time_window_hours)

@router.websocket("/monitoring/kpis/stream")
async def stream_kpis(websocket: WebSocket, handler: MonitoringHandler = Depends(lambda: MonitoringHandler(DatabaseConfig())), user_id: str = Depends(lambda x: x.query_params.get("user_id"))):
    if not user_id:
        await websocket.close(code=1008, reason="Missing user_id")
        return
    await handler.stream_kpis(websocket, user_id)
