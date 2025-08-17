from fastapi import FastAPI
from .routes import wallet, oauth, troubleshoot, quantum_link, mcp_protocol, credentials, void
from .health import router as health_router
from ..utils.logging import log_error, log_info
from ..utils.middleware import add_middleware
from ..providers import anthropic_provider, openai_provider, xai_provider, google_provider
import asyncio
import torch
import tensorflow as tf
import dspy
from datetime import datetime

app = FastAPI(
    title="WEBXOS MCP Gateway",
    description="API Gateway for WEBXOS Vial MCP (BETA) with PyTorch, TensorFlow, and DSPy",
    version="2.7.8",
    openapi_url="/v1/openapi.json"
)

add_middleware(app)

app.include_router(wallet.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
app.include_router(troubleshoot.router, prefix="/v1")
app.include_router(quantum_link.router, prefix="/v1")
app.include_router(mcp_protocol.router, prefix="/v1")
app.include_router(credentials.router, prefix="/v1")
app.include_router(void.router, prefix="/v1")
app.include_router(health_router, prefix="/v1")
app.include_router(anthropic_provider.router, prefix="/v1")
app.include_router(openai_provider.router, prefix="/v1")
app.include_router(xai_provider.router, prefix="/v1")
app.include_router(google_provider.router, prefix="/v1")

dspy.settings.configure(lm=dspy.LM('gpt-3.5-turbo'))

@app.on_event("startup")
async def startup_event():
    log_info("WEBXOS MCP Gateway starting...")
    log_info(f"PyTorch version: {torch.__version__}")
    log_info(f"TensorFlow version: {tf.__version__}")
    log_info(f"DSPy configured with: {dspy.settings.lm}")

@app.on_event("shutdown")
async def shutdown_event():
    log_info("WEBXOS MCP Gateway shutting down...")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log_error(f"Traceback: Global error: {str(exc)}")
    return {"error": "Internal Server Error", "detail": str(exc)}, 500
