from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from config.config import DatabaseConfig, APIConfig
from tools.auth_tool import AuthTool
from tools.vial_management import VialManagementTool
from tools.wallet import WalletTool
from lib.security import SecurityHandler
from lib.mcp_transport import MCPTransport
from lib.logger import logger
from lib.errors import ValidationError
from neondatabase import AsyncClient
import redis.asyncio as redis
import os
import re
from html import escape

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
db_client = AsyncClient(DatabaseConfig().database_url)
auth_tool = AuthTool(db_client)
vial_tool = VialManagementTool(db_client)
wallet_tool = WalletTool(db_client)
security_handler = SecurityHandler(db_client)
mcp_transport = MCPTransport()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-Session-ID", "Content-Type"]
)

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: int

def sanitize_input(value: Any) -> Any:
    """Sanitize input to prevent injection attacks."""
    if isinstance(value, str):
        # Remove potentially malicious characters and escape HTML
        value = re.sub(r'[<>;{}]', '', value)
        return escape(value)
    elif isinstance(value, dict):
        return {k: sanitize_input(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_input(item) for item in value]
    return value

async def get_current_user(token: str = Depends(oauth2_scheme), session_id: str = Depends(lambda x: x.headers.get("X-Session-ID"))):
    user = await auth_tool.verify_token(token, session_id)
    if not user:
        await security_handler.log_event(
            event_type="unauthorized_access",
            user_id=None,
            details={"token": token[:8] + "...", "session_id": session_id}
        )
        raise HTTPException(status_code=401, detail="Invalid token or session")
    
    await security_handler.enforce_concurrent_session_limit(user["user_id"])
    return user

@app.on_event("startup")
async def startup_event():
    await db_client.connect()
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    await FastAPILimiter.init(redis_client)
    logger.info("Connected to Neon Postgres database and initialized rate limiter")

@app.on_event("shutdown")
async def shutdown_event():
    await db_client.disconnect()
    await FastAPILimiter.close()
    logger.info("Disconnected from Neon Postgres database and closed rate limiter")

@app.post("/mcp/execute", response_model=JSONRPCRequest, dependencies=[Depends(RateLimiter(times=100, seconds=900))])
async def execute(request: JSONRPCRequest, user: Dict[str, Any] = Depends(get_current_user)):
    try:
        # Sanitize request parameters
        sanitized_params = sanitize_input(request.params)
        method = request.method
        params = sanitized_params
        params["user_id"] = user["user_id"]
        
        # Apply stricter rate limiting for cash-out
        if method == "wallet.cashOut":
            await RateLimiter(times=5, seconds=900)(request)
        
        if method.startswith("auth."):
            result = await auth_tool.execute(params)
        elif method.startswith("vial_management."):
            result = await vial_tool.execute(params)
        elif method.startswith("wallet."):
            result = await wallet_tool.execute(params)
        else:
            raise ValidationError(f"Unknown method: {method}")
        
        await security_handler.log_event(
            event_type="api_request",
            user_id=user["user_id"],
            details={"method": method, "params": {k: v for k, v in params.items() if k != "code"}}
        )
        return JSONRPCResponse(jsonrpc="2.0", result=result, id=request.id)
    except Exception as e:
        logger.error(f"Error executing {method}: {str(e)}")
        await security_handler.log_event(
            event_type="api_error",
            user_id=user.get("user_id"),
            details={"method": method, "error": str(e)}
        )
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
