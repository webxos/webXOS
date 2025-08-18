from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import config
from .auth import handle_auth
from .wallet import validate_wallet
from .agents import handle_command, configure_compute, refresh_configuration, terminate_fast, terminate_immediate
from .database import get_db
from .monitoring import get_status
import asyncpg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mcp/api/endpoints")
async def endpoints(request: dict, db=Depends(get_db)):
    method = request.get("method")
    if method == "authenticate":
        return await handle_auth(method, request, db)
    elif method == "validate_md_wallet":
        return await validate_wallet(request.get("wallet_data"), db)
    elif method == "generate_api_key":
        from .auth import generate_api_key
        return await generate_api_key(request.get("user_id"), db)
    elif method == "configure":
        return await configure_compute(request.get("spec", {}), db)
    elif method == "refresh_configuration":
        return await refresh_configuration(db)
    elif method == "terminate_fast":
        return await terminate_fast(db)
    elif method == "terminate_immediate":
        return await terminate_immediate(db)
    elif request.get("command"):
        return await handle_command(request.get("command"), request, db)
    raise HTTPException(status_code=400, detail="Invalid method or command")

@app.get("/mcp/api/status")
async def status(db=Depends(get_db)):
    return await get_status(db)

@app.post("/mcp/api/relay_signal")
async def relay_signal():
    return {"status": "signal_relayed"}

# xAI Artifact Tags: #vial2 #fastapi #backend #neon_mcp
