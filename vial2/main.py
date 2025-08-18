from fastapi import FastAPI
from .api.endpoints import router as endpoints_router
from .api.jsonrpc import router as jsonrpc_router
from .api.git_commands import router as git_router
from .api.wallet_ops import router as wallet_router
from .api.quantum_link import router as quantum_router
from .monitoring.health import router as health_router
from .monitoring.log_aggregation import router as logs_router
from .monitoring.alerts import router as alerts_router
from .security.cors import configure_cors
from .logging import configure_logging
from .api.http_transport import configure_http_transport
from .error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS, HTTP transport, and logging
configure_cors(app)
configure_http_transport(app)
configure_logging()

# Include routers
app.include_router(endpoints_router)
app.include_router(jsonrpc_router)
app.include_router(git_router)
app.include_router(wallet_router)
app.include_router(quantum_router)
app.include_router(health_router)
app.include_router(logs_router)
app.include_router(alerts_router)

@app.on_event("startup")
async def startup_event():
    try:
        from .config import config
        config.validate()
        logger.info("Application started successfully")
    except Exception as e:
        error_logger.log_error("startup", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params=None)
        logger.error(f"Startup failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #main #sqlite #octokit #neon_mcp
