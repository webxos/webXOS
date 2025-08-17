from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from socketio import AsyncServer, ASGIApp
from routes import health, oauth, wallet, credentials, void, troubleshoot, quantum_link, git
from mcp.server import MCPServer

app = FastAPI()
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app.mount("/socket.io", ASGIApp(sio))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
app.include_router(wallet.router, prefix="/v1")
app.include_router(credentials.router, prefix="/v1")
app.include_router(void.router, prefix="/v1")
app.include_router(troubleshoot.router, prefix="/v1")
app.include_router(quantum_link.router, prefix="/v1")
app.include_router(git.router, prefix="/v1")
app.include_router(MCPServer().router, prefix="/v1")

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def update(sid, data):
    await sio.emit('update', data)
