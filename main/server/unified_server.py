from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from .mcp.mcp_server_auth import MCPAuthHandler, AuthRequest, RefreshRequest
from .mcp.mcp_server_quantum import MCPQuantumHandler, QuantumRequest
from .mcp.mcp_server_notes import MCPNotesHandler, NoteRequest, NoteReadRequest
from .mcp.mcp_server_resources import MCPResourcesHandler, ResourceRequest
from .mcp.webxos_wallet import MCPWalletManager, WalletRequest
from .mcp.health_check import add_health_check
from .mcp.db.db_manager import DatabaseManager
from .mcp.base_prompt import BasePromptManager
import os
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Database configurations
postgres_config = {
    "host": os.getenv("POSTGRES_HOST", "postgresdb"),
    "port": int(os.getenv("POSTGRES_DOCKER_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "database": os.getenv("POSTGRES_DB", "vial_mcp")
}
mysql_config = {
    "host": os.getenv("MYSQL_HOST", "mysqldb"),
    "port": int(os.getenv("MYSQL_DOCKER_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_ROOT_PASSWORD", "mysql"),
    "database": os.getenv("MYSQL_DB", "vial_mcp")
}
mongo_config = {
    "host": os.getenv("MONGO_HOST", "mongodb"),
    "port": int(os.getenv("MONGO_DOCKER_PORT", 27017)),
    "username": os.getenv("MONGO_USER", "mongo"),
    "password": os.getenv("MONGO_PASSWORD", "mongo")
}

db_manager = DatabaseManager(postgres_config, mysql_config, mongo_config)
db_manager.connect()
auth_handler = MCPAuthHandler()
quantum_handler = MCPQuantumHandler()
notes_handler = MCPNotesHandler()
resources_handler = MCPResourcesHandler()
wallet_manager = MCPWalletManager()
prompt_manager = BasePromptManager()

add_health_check(app)

@app.post("/api/auth/login")
async def login(request: AuthRequest):
    return await auth_handler.authenticate(request)

@app.post("/api/auth/refresh")
async def refresh(request: RefreshRequest):
    return await auth_handler.refresh_token(request)

@app.post("/api/quantum/link")
async def quantum_link(request: QuantumRequest, access_token: str = Depends(lambda x: x)):
    return await quantum_handler.process_quantum(request, access_token)

@app.post("/api/notes/add")
async def add_note(request: NoteRequest, access_token: str = Depends(lambda x: x)):
    payload = auth_handler.auth_manager.verify_token(access_token)
    if payload["wallet_id"] != request.wallet_id:
        raise HTTPException(status_code=401, detail="Unauthorized wallet access")
    return db_manager.add_note(request.wallet_id, request.content, request.resource_id, request.db_type)

@app.post("/api/notes/read")
async def read_note(request: NoteReadRequest, access_token: str = Depends(lambda x: x)):
    payload = auth_handler.auth_manager.verify_token(access_token)
    if payload["wallet_id"] != request.wallet_id:
        raise HTTPException(status_code=401, detail="Unauthorized wallet access")
    return db_manager.get_notes(request.wallet_id, 10, request.db_type)

@app.post("/api/resources/latest")
async def get_resources(request: ResourceRequest, access_token: str = Depends(lambda x: x)):
    return await resources_handler.get_latest_resources(request, access_token)

@app.post("/api/wallet/create")
async def create_wallet(request: WalletRequest, access_token: str = Depends(lambda x: x)):
    return await wallet_manager.create_wallet(request, access_token)

@app.post("/api/prompts/get")
async def get_prompt(agent_name: str, wallet_id: str = None, vial_id: str = None, content: str = None):
    return {"prompt": prompt_manager.get_prompt(agent_name, wallet_id, vial_id, content)}
