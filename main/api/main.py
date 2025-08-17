from fastapi import FastAPI
from .routes import wallet, oauth, troubleshoot, quantum_link, mcp_endpoints
from .health import router as health_router
from ..utils.logging import log_error, log_info
import asyncio

app = FastAPI(
    title="WEBXOS MCP Gateway",
    description="API Gateway for WEBXOS Vial MCP (BETA)",
    version="2.7.6",
    openapi_url="/v1/openapi.json"
)

# Include routers
app.include_router(wallet.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
app.include_router(troubleshoot.router, prefix="/v1")
app.include_router(quantum_link.router, prefix="/v1")
app.include_router(mcp_endpoints.router, prefix="/v1")
app.include_router(health_router, prefix="/v1")

# Startup event
@app.on_event("startup")
async def startup_event():
    log_info("WEBXOS MCP Gateway starting...")
    # Add database or resource initialization here

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    log_info("WEBXOS MCP Gateway shutting down...")
    # Add cleanup logic here

# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log_error(f"Traceback: Global error: {str(exc)}")
    return {"error": "Internal Server Error", "detail": str(exc)}, 500
