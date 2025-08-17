from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict, Any, List
from config.config import DatabaseConfig, APIConfig
from tools.auth_tool import AuthTool
from tools.vial_management import VialManagementTool
from tools.wallet import WalletTool
from lib.mcp_transport import MCPTransport
from lib.logger import logger
from lib.errors import ValidationError
from neondatabase import AsyncClient
import json
import os

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
db_client = AsyncClient(DatabaseConfig().database_url)
auth_tool = AuthTool(db_client)
vial_tool = VialManagementTool(db_client)
wallet_tool = WalletTool(db_client)
mcp_transport = MCPTransport()

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: int

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Any = None
    id: int

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = await auth_tool.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.on_event("startup")
async def startup_event():
    await db_client.connect()
    logger.info("Connected to Neon Postgres database")

@app.on_event("shutdown")
async def shutdown_event():
    await db_client.disconnect()
    logger.info("Disconnected from Neon Postgres database")

@app.post("/mcp/execute", response_model=JSONRPCResponse)
async def execute(request: JSONRPCRequest, user: Dict[str, Any] = Depends(get_current_user)):
    try:
        method = request.method
        params = request.params
        params["user_id"] = user["user_id"]
        
        if method.startswith("auth."):
            result = await auth_tool.execute(params)
        elif method.startswith("vial_management."):
            result = await vial_tool.execute(params)
        elif method.startswith("wallet."):
            result = await wallet_tool.execute(params)
        else:
            raise ValidationError(f"Unknown method: {method}")
        
        return JSONRPCResponse(jsonrpc="2.0", result=result, id=request.id)
    except Exception as e:
        logger.error(f"Error executing {method}: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            error={"code": -32603, "message": str(e)},
            id=request.id
        )

@app.get("/openapi.json")
async def get_openapi():
    return app.openapi()

@app.get("/docs")
async def get_docs():
    return app.openapi()
