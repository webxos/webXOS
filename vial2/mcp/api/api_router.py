from fastapi import APIRouter
from .auth_relay import router as auth_relay_router
from .console_commands import router as console_router
from .error_traceback import router as error_traceback_router
from .wallet_sync import router as wallet_sync_router
from .quantum_link import router as quantum_link_router
from .api_key_generate import router as api_key_generate_router
from .mining_verification import router as mining_verification_router
from .wallet_export_import import router as wallet_export_import_router
from ...error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

api_router = APIRouter()

api_router.include_router(auth_relay_router, prefix="/auth", tags=["auth"])
api_router.include_router(console_router, prefix="/console", tags=["console"])
api_router.include_router(error_traceback_router, prefix="/error", tags=["error"])
api_router.include_router(wallet_sync_router, prefix="/wallet", tags=["wallet"])
api_router.include_router(quantum_link_router, prefix="/quantum", tags=["quantum"])
api_router.include_router(api_key_generate_router, prefix="/api_key", tags=["api_key"])
api_router.include_router(mining_verification_router, prefix="/mining", tags=["mining"])
api_router.include_router(wallet_export_import_router, prefix="/wallet", tags=["wallet"])

try:
    logger.info("API router initialized with all endpoints")
except Exception as e:
    error_logger.log_error("api_router_init", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
    logger.error(f"API router initialization failed: {str(e)}")

# xAI Artifact Tags: #vial2 #mcp #api #router #neon_mcp
