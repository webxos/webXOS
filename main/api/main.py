from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from .routes.wallet import router as wallet_router
from .routes.oauth import router as oauth_router
from .routes.git import router as git_router
from .routes.troubleshoot import router as troubleshoot_router
from .routes.quantum_link import router as quantum_link_router
from .routes.mcp_protocol import router as mcp_protocol_router
from .routes.credentials import router as credentials_router
from .routes.void import router as void_router
from .health import router as health_router
from .mcp.transport.http import setup_http_transport
from .mcp.transport.sse import router as sse_router
from .providers import anthropic_provider, openai_provider, xai_provider, google_provider
from .mcp.server import MCPServer
from ..utils.logging import log_error, log_info
from ..utils.authentication import verify_token
import asyncio
import torch
import tensorflow as tf
import dspy

app = FastAPI(
    title="WEBXOS MCP Gateway",
    description="API Gateway for WEBXOS Vial MCP with Jupyter, NLP, and Git integration",
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

# Initialize MCP server and Socket.IO
mcp_server = MCPServer()
sio = SocketManager(app=app)

# Include routers with authentication dependency
protected_routers = [
    wallet_router,
    git_router,
    troubleshoot_router,
    quantum_link_router,
    mcp_protocol_router,
    credentials_router,
    void_router,
    anthropic_provider.router,
    openai_provider.router,
    xai_provider.router,
    google_provider.router,
    setup_http_transport(mcp_server),
    sse_router
]
for router in protected_routers:
    app.include_router(router, prefix="/v1", dependencies=[Depends(verify_token)])
app.include_router(oauth_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")

# Socket.IO events
@sio.on('connect')
async def handle_connect(sid, environ):
    log_info(f"WebSocket client connected: {sid}")

@sio.on('disconnect')
async def handle_disconnect(sid):
    log_info(f"WebSocket client disconnected: {sid}")

async def broadcast_updates():
    while True:
        try:
            await sio.emit('update', {
                "balance": 0.0,
                "reputation": 0,
                "task_status": "Idle"
            })
            await mcp_server.notify_message("Server heartbeat")
            await asyncio.sleep(30)
        except Exception as e:
            log_error(f"WebSocket broadcast error: {str(e)}")

dspy.settings.configure(lm=dspy.LM('gpt-3.5-turbo'))

@app.on_event("startup")
async def startup_event():
    log_info("WEBXOS MCP Gateway starting...")
    log_info(f"PyTorch version: {torch.__version__}")
    log_info(f"TensorFlow version: {tf.__version__}")
    log_info(f"DSPy configured with: {dspy.settings.lm}")
    asyncio.create_task(broadcast_updates())

@app.on_event("shutdown")
async def shutdown_event():
    log_info("WEBXOS MCP Gateway shutting down...")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log_error(f"Traceback: Global error: {str(exc)}")
    return {"error": "Internal Server Error", "detail": str(exc)}, 500
