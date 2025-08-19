from fastapi import FastAPI
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.api import agent_endpoint, auth_endpoint, health_endpoint, wallet_sync
from mcp.database import neon_connection
import uvicorn

app = FastAPI()
mcp_server = Server(InitializationOptions(tools=["/vial/train", "/vial/sync", "/vial/quantum"], resources=["vial://config", "vial://wallet", "vial://status"]))

app.include_router(agent_endpoint.router, prefix="/mcp/api/vial")
app.include_router(auth_endpoint.router, prefix="/mcp/api/vial")
app.include_router(health_endpoint.router, prefix="/mcp/api/vial")
app.include_router(wallet_sync.router, prefix="/mcp/api/vial")

@app.on_event("startup")
async def startup_event():
    await neon_connection.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await neon_connection.disconnect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# xAI Artifact Tags: #vial2 #main #mcp #server #neon_mcp
