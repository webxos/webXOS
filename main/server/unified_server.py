# main/server/unified_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .mcp.api_gateway.gateway_router import router as gateway_router
from .mcp.auth.auth_manager import AuthManager
from .mcp.notes.mcp_server_notes import NotesService
from .mcp.quantum.quantum_simulator import QuantumSimulator
from .mcp.agents.global_mcp_agents import GlobalMCPAgents
from .mcp.agents.translator_agent import TranslatorAgent
from .mcp.agents.library_agent import LibraryAgent
from .mcp.wallet.webxos_wallet import WalletService
from .mcp.utils.health_check import router as health_router
from .mcp.utils.mcp_error_handler import handle_mcp_error
import logging
import logging.config
import os

app = FastAPI(title="Vial MCP Controller", version="1.0")

# Load logging configuration
logging.config.fileConfig('main/server/mcp/config/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger("mcp")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gateway_router, prefix="/mcp")
app.include_router(health_router)

# Services
auth_manager = AuthManager()
notes_service = NotesService()
quantum_simulator = QuantumSimulator()
global_agents = GlobalMCPAgents()
translator_agent = TranslatorAgent()
library_agent = LibraryAgent()
wallet_service = WalletService()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Vial MCP Controller")
    # Initialize services (e.g., check DB connections)
    try:
        await notes_service.collection.find_one()
        await wallet_service.verify_wallet("0x0000000000000000000000000000000000000000")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Vial MCP Controller")
    notes_service.close()
    global_agents.close()
    translator_agent.close()
    library_agent.close()

@app.exception_handler(MCPError)
async def mcp_error_handler(request, exc: MCPError):
    return handle_mcp_error(exc)

@app.get("/")
async def root():
    return {"message": "Vial MCP Controller API", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"), port=int(os.getenv("MCP_SERVER_PORT", 8080)))
