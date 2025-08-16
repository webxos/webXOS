from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from starlette.middleware.cors import CORS
from starlette.responses import StreamingResponse
from .routes import wallet, oauth, troubleshoot, quantum_link, mcp_endpoints
from ..security.authentication import verify_token
from ..utils.logging import log_error, log_info
from ..config.redis_config import get_redis
from .mcp_protocol import router as mcp_router
import asyncio
import json

app = FastAPI(
    title="WEBXOS MCP Gateway",
    description="API Gateway for WEBXOS Vial MCP (BETA)",
    version="2.7.0",
    openapi_url="/v1/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORS,
    allow_origins=["https://api.webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(wallet.router, dependencies=[Depends(verify_token)])
app.include_router(oauth.router)
app.include_router(troubleshoot.router, dependencies=[Depends(verify_token)])
app.include_router(quantum_link.router, dependencies=[Depends(verify_token)])
app.include_router(mcp_endpoints.router, dependencies=[Depends(verify_token)])
app.include_router(mcp_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Server-Sent Events for progress tracking
@app.get("/v1/progress")
async def progress_stream():
    async def event_generator():
        for i in range(1, 101, 10):
            yield f"data: {json.dumps({'progress': i, 'status': 'processing'})}\n\n"
            await asyncio.sleep(1)
        yield f"data: {json.dumps({'progress': 100, 'status': 'completed'})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Provider failover event handler
@local_handler.register(event_name="provider_error")
async def handle_provider_error(event: Event, redis=Depends(get_redis)):
    provider = event.payload.get("provider")
    error = event.payload.get("error")
    log_error(f"Provider {provider} failed: {error}")
    await redis.set(f"provider:{provider}:status", "unavailable", ex=3600)
    return {"status": "failover_triggered", "provider": provider}

# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log_error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    log_info("WEBXOS MCP Gateway starting...")
    redis = await get_redis()
    await redis.set("system:status", "running")
