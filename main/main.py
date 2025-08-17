from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main.api.mcp.server import MCPServer
from main.api.utils.logging import setup_logging

app = FastAPI(title="Neon MCP API", version="3.0.0")

# Setup logging
setup_logging()

# CORS configuration for Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP server
mcp_server = MCPServer()

@app.get("/api/v1/health")
async def health():
    return mcp_server.health_check()

@app.post("/api/v1/oauth/token")
async def token(request: dict):
    return await mcp_server.authenticate(request)

@app.post("/api/v1/train/{vial_id}")
async def train(vial_id: str, dataset: dict):
    return await mcp_server.train_vial(vial_id, dataset)

@app.post("/api/v1/void")
async def void():
    return await mcp_server.void()

@app.get("/api/v1/troubleshoot")
async def troubleshoot():
    return await mcp_server.troubleshoot()

@app.post("/api/v1/quantum_link")
async def quantum_link():
    return await mcp_server.quantum_link()

@app.get("/api/v1/credentials")
async def credentials():
    return await mcp_server.get_credentials()

@app.get("/")
async def root():
    return {"message": "Neon MCP API - Use /api/v1/health to check status"}
