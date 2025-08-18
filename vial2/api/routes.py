from fastapi import APIRouter, Depends, HTTPException
from ..database import get_db
from ..auth import handle_auth, generate_api_key
from ..wallet import validate_wallet
from ..agents import handle_command, configure_compute, refresh_configuration, terminate_fast, terminate_immediate
from ..monitoring import get_status, replication_status
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/api", tags=["mcp"])

@router.post("/endpoints")
async def endpoints(request: dict, db=Depends(get_db)):
    try:
        method = request.get("method")
        if method == "authenticate":
            return await handle_auth(method, request)
        elif method == "validate_md_wallet":
            return await validate_wallet(request.get("wallet_data"), db)
        elif method == "generate_api_key":
            return await generate_api_key(request.get("user_id"))
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
    except Exception as e:
        error_logger.log_error("api_routes", f"Endpoint processing failed: {str(e)}", str(e.__traceback__))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
async def status(db=Depends(get_db)):
    try:
        return await get_status(db)
    except Exception as e:
        error_logger.log_error("api_routes", f"Status fetch failed: {str(e)}", str(e.__traceback__))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/replication_status")
async def get_replication_status(db=Depends(get_db)):
    try:
        return await replication_status(db)
    except Exception as e:
        error_logger.log_error("api_routes", f"Replication status fetch failed: {str(e)}", str(e.__traceback__))
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/relay_signal")
async def relay_signal():
    try:
        return {"status": "signal_relayed"}
    except Exception as e:
        error_logger.log_error("api_routes", f"Relay signal failed: {str(e)}", str(e.__traceback__))
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #routes #neon_mcp
