from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict, Any
import asyncpg
from .config import config
from .tools import auth_tool, wallet, vial_management, git_tool
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="Vial MCP v2 API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/vial2/mcp/api/auth_tool")

class CommandRequest(BaseModel):
    command: str
    method: str | None = None
    wallet_data: Dict[str, Any] | None = None
    spec: Dict[str, Any] | None = None
    user_id: str | None = None

async def get_db():
    conn = await asyncpg.connect(config.DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

@app.on_event("startup")
async def startup():
    logger.info("Starting Vial MCP v2 API")
    await auth_tool.init_auth()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/auth_tool")
async def auth_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await auth_tool.handle_auth(request.method, request, db)
        return result
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/wallet/validate")
async def validate_wallet(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await wallet.validate_wallet(request.wallet_data, db)
        return result
    except Exception as e:
        logger.error(f"Wallet validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management")
async def vial_management_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.handle_command(request.command, request, db)
        return result
    except Exception as e:
        logger.error(f"Vial management error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management/configure")
async def configure_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.configure_compute(request.spec, db)
        return result
    except Exception as e:
        logger.error(f"Configure error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management/refresh_configuration")
async def refresh_config_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.refresh_configuration(db)
        return result
    except Exception as e:
        logger.error(f"Refresh config error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management/terminate_fast")
async def terminate_fast_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.terminate_fast(db)
        return result
    except Exception as e:
        logger.error(f"Terminate fast error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management/terminate_immediate")
async def terminate_immediate_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.terminate_immediate(db)
        return result
    except Exception as e:
        logger.error(f"Terminate immediate error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial_management/troubleshoot")
async def troubleshoot_endpoint(db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await vial_management.troubleshoot(db)
        return result
    except Exception as e:
        logger.error(f"Troubleshoot error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tools/git_tool")
async def git_endpoint(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await git_tool.execute_git_command(request.command, db)
        return result
    except Exception as e:
        logger.error(f"Git command error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth_tool/generate_api_key")
async def generate_api_key(request: CommandRequest, db: asyncpg.Connection = Depends(get_db)):
    try:
        result = await auth_tool.generate_api_key(request.user_id, db)
        return result
    except Exception as e:
        logger.error(f"API key generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #fastapi #neon_mcp #oauth2
