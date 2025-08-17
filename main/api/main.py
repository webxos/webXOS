from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.wallet import router as wallet_router
from .routes.oauth import router as oauth_router
from .routes.troubleshoot import router as troubleshoot_router
from .routes.quantum_link import router as quantum_link_router
from .routes.mcp_protocol import router as mcp_protocol_router
from .routes.credentials import router as credentials_router
from .routes.void import router as void_router
from .health import router as health_router
from .providers import anthropic_provider, openai_provider, xai_provider, google_provider
from ..utils.logging import log_error, log_info
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wallet_router, prefix="/v1")
app.include_router(oauth_router, prefix="/v1")
app.include_router(troubleshoot_router, prefix="/v1")
app.include_router(quantum_link_router, prefix="/v1")
app.include_router(mcp_protocol_router, prefix="/v1")
app.include_router(credentials_router, prefix="/v1")
app.include_router(void_router, prefix="/v1")
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
