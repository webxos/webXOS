from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .api import agent_api, auth_api, health_api, json_api, wallet_sync_api
from .database import get_db, get_sqlite_db
from .security import oauth_auth
from pydantic import BaseModel

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    command: str

# JSON-RPC 2.0 Compliance
@app.post("/mcp/api/command")
async def process_command(request: CommandRequest, auth=Depends(oauth_auth)):
    try:
        db = await get_db()
        async with db.acquire() as conn:
            result = await conn.fetchval("SELECT process_command($1)", request.command)
        return {"jsonrpc": "2.0", "result": result, "id": "1"}
    except Exception as e:
        return {"jsonrpc": "2.0", "error": str(e), "id": "1"}

@app.post("/mcp/tools/call")
async def call_tool(tool: str, auth=Depends(oauth_auth)):
    return {"jsonrpc": "2.0", "result": {"tool": tool}, "id": "1"}

@app.get("/mcp/api/health")
async def health_check(auth=Depends(oauth_auth)):
    sqlite_db = await get_sqlite_db()
    return {"jsonrpc": "2.0", "result": {"status": "healthy"}, "id": "1"}

# Include API endpoints
app.include_router(agent_api)
app.include_router(auth_api)
app.include_router(health_api)
app.include_router(json_api)
app.include_router(wallet_sync_api)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
