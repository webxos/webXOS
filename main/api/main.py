from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from starlette.middleware.cors import CORS
from starlette.responses import StreamingResponse
from .routes import wallet, oauth, troubleshoot, quantum_link, mcp_endpoints
from ..security.authentication import verify_token
from ..utils.logging import log_error, log_info
from ..config.redis_config import get_redis
from ..resources.database_resources import db_resources
from .mcp_protocol import router as mcp_router
from .middleware.auth import auth_middleware
from .middleware.cors import cors_middleware
from .middleware.rate_limit import rate_limit_middleware
from .middleware.logging import logging_middleware
from ..providers.base_provider import BaseProvider
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_provider import OpenAIProvider
import asyncio
import json

app = FastAPI(
    title="WEBXOS MCP Gateway",
    description="API Gateway for WEBXOS Vial MCP (BETA)",
    version="2.7.1",
    openapi_url="/v1/openapi.json"
)

# Initialize providers
providers = {
    "anthropic": AnthropicProvider("anthropic_api_key"),
    "openai": OpenAIProvider("openai_api_key")
}

# Apply middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(cors_middleware)
app.middleware("http")(auth_middleware)

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
    available_provider = next((p for p in providers.values() if p.is_available), None)
    if available_provider:
        await available_provider.check_availability()
        return {"status": "failover_triggered", "provider": available_provider.__class__.__name__}
    return {"status": "no_fallback_available"}

# MCP interceptor
@app.middleware("http")
async def mcp_interceptor(request: Request, call_next):
    if request.url.path.startswith("/v1/mcp"):
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            log_error("MCP request without authentication")
            raise HTTPException(status_code=401, detail="Authentication required")
        provider = providers.get("openai")  # Default to OpenAI, add dynamic selection later
        if provider and provider.is_available:
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                await handle_provider_error(Event(name="provider_error", payload={"provider": "openai", "error": str(e)}))
                raise
        else:
            log_error("No available provider for MCP request")
            raise HTTPException(status_code=503, detail="Service unavailable")
    response = await call_next(request)
    return response

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
    await db_resources.connect()
    redis = await get_redis()
    await redis.set("system:status", "running")
    for provider in providers.values():
        await provider.check_availability()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    log_info("WEBXOS MCP Gateway shutting down...")
    await db_resources.disconnect()
