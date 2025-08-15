# main/server/mcp/utils/rate_limiter.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis
import time
import hashlib
from typing import Optional
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
import os

class RateLimiter:
    def __init__(self, app: FastAPI):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.metrics = PerformanceMetrics()
        self.default_limits = {"minute": 100, "hour": 1000}
        self.adaptive_threshold = 0.9  # Trigger adaptive limits at 90% of quota
        app.middleware("http")(self.rate_limit_middleware)

    def get_user_tier(self, request: Request) -> str:
        # Placeholder for user tier detection (e.g., free, premium)
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return "anonymous"
        try:
            payload = self.metrics.verify_token(token)
            return payload.get("tier", "free")
        except:
            return "anonymous"

    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        with self.metrics.track_span("check_rate_limit", {"key": key}):
            try:
                current = int(time.time())
                window_key = f"{key}:{current // window}"
                count = self.redis_client.get(window_key) or 0
                count = int(count)
                if count >= limit:
                    return False
                self.redis_client.incr(window_key)
                self.redis_client.expire(window_key, window)
                return True
            except Exception as e:
                handle_generic_error(e, context="check_rate_limit")
                return True  # Fallback to allow request if Redis fails

    async def analyze_behavior(self, request: Request) -> float:
        # Placeholder for AI-powered abuse detection
        # In a full implementation, analyze request patterns (e.g., frequency, payload size)
        return 0.5  # Mock risk score

    async def rate_limit_middleware(self, request: Request, call_next):
        with self.metrics.track_span("rate_limit_middleware", {}):
            try:
                key = f"{request.client.host}:{self.get_user_tier(request)}"
                risk_score = await self.analyze_behavior(request)
                
                # Adjust limits based on risk score
                minute_limit = int(self.default_limits["minute"] * (1 - risk_score * self.adaptive_threshold))
                hour_limit = int(self.default_limits["hour"] * (1 - risk_score * self.adaptive_threshold))

                # Check rate limits
                if not await self.check_rate_limit(key, minute_limit, 60):
                    raise HTTPException(429, detail="Rate limit exceeded (minute)")
                if not await self.check_rate_limit(key, hour_limit, 3600):
                    raise HTTPException(429, detail="Rate limit exceeded (hour)")

                response = await call_next(request)
                return response
            except HTTPException as e:
                self.metrics.record_error("rate_limit_exceeded", str(e))
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
            except Exception as e:
                handle_generic_error(e, context="rate_limit_middleware")
                return JSONResponse(status_code=500, content={"detail": "Internal server error"})
