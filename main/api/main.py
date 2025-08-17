from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from .routes import oauth, wallet, git, troubleshoot, quantum_link, mcp_protocol, credentials, void
from .health import router as health_router
from .mcp.transport.http import router as http_router
from .mcp.transport.sse import sio
from ..utils.monitoring import start_monitoring

app = FastAPI(title="WEBXOS MCP Gateway", version="2.7.8")
sio.mount(app, socketio_path="/ws")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(oauth.router, prefix="/v1")
app.include_router(wallet.router, prefix="/v1")
app.include_router(git.router, prefix="/v1")
app.include_router(troubleshoot.router, prefix="/v1")
app.include_router(quantum_link.router, prefix="/v1")
app.include_router(mcp_protocol.router, prefix="/v1")
app.include_router(credentials.router, prefix="/v1")
app.include_router(void.router, prefix="/v1")
app.include_router(health_router, prefix="/v1")
app.include_router(http_router, prefix="/v1")

@app.on_event("startup")
async def startup_event():
    await start_monitoring()
