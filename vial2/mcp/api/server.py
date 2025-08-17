from fastapi import FastAPI, HTTPException, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.config import DatabaseConfig
from tools.auth_tool import AuthTool
from tools.wallet import WalletTool
from tools.vial_management import VialManager
from tools.git_tool import GitTool
from lib.security import SecurityHandler
from lib.notifications import NotificationHandler
from sql.query_engine import QueryEngine
from monitoring.metrics import MetricsCollector
from postgrest import AsyncPostgrestClient
import logging
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vial MCP API",
    description="API for Vial MCP with Neon, PyTorch, and Stack Auth integration",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "endpoints", "description": "Command processing"},
        {"name": "vials", "description": "Vial management"},
        {"name": "wallet", "description": "Wallet operations"},
        {"name": "ws", "description": "WebSocket notifications"}
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DatabaseConfig()
auth_tool = AuthTool(db)
wallet_tool = WalletTool(db)
vial_manager = VialManager(db)
security = SecurityHandler(db)
notifications = NotificationHandler()
query_engine = QueryEngine(db)
git_tool = GitTool(db, security)
metrics = MetricsCollector()
data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")

@app.on_event("startup")
async def startup():
    try:
        await db.connect()
        logger.info(f"Connected to database: {db.url} [server.py:45] [ID:startup_success]")
        metrics.increment("startup_success")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)} [server.py:50] [ID:startup_error]")
        metrics.increment("startup_error")
        raise

@app.on_event("shutdown")
async def shutdown():
    try:
        await db.disconnect()
        await data_api.aclose()
        logger.info("Database and Data API connections closed [server.py:55] [ID:shutdown_success]")
        metrics.increment("shutdown_success")
    except Exception as e:
        logger.error(f"Shutdown failed: {str(e)} [server.py:60] [ID:shutdown_error]")
        metrics.increment("shutdown_error")
        raise

@app.post("/vial2/mcp/api/auth", tags=["auth"])
async def auth_endpoint(data: dict):
    try:
        result = await auth_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "auth", data)
        metrics.increment("auth_requests")
        return result
    except Exception as e:
        error_message = f"Auth failed: {str(e)} [server.py:70] [ID:auth_error]"
        await security.log_error("unknown", "auth", error_message)
        metrics.increment("auth_errors")
        raise HTTPException(status_code=400, detail=error_message)

@app.post("/vial2/mcp/api/endpoints", tags=["endpoints"])
async def command_endpoint(data: dict):
    try:
        user_id = data.get("user_id")
        command = data.get("command")
        project_id = data.get("project_id", "twilight-art-21036984")
        metrics.increment(f"command_{command}_requests")
        if command == "data":
            table = data.get("args")[0]
            action = data.get("args")[1]
            if table == "sql":
                result = await query_engine.execute_sql(user_id, " ".join(data.get("args")[1:]), data.get("access_token"), project_id)
            else:
                result = await query_engine.execute_data_api_query(user_id, table, action, data.get("access_token"), project_id)
        elif command == "git":
            result = await git_tool.execute(data)
        elif command in ["prompt", "task", "config"]:
            result = await vial_manager.execute(data)
        else:
            error_message = f"Invalid command: {command} [server.py:85] [ID:command_error]"
            metrics.increment("command_errors")
            raise ValueError(error_message)
        await security.log_action(user_id, command, data)
        return result
    except Exception as e:
        error_message = f"Command failed: {str(e)} [server.py:90] [ID:command_error]"
        await security.log_error(user_id or "unknown", "command", error_message)
        metrics.increment("command_errors")
        raise HTTPException(status_code=400, detail=error_message)

@app.get("/vial2/mcp/api/health", tags=["endpoints"])
async def health_check():
    try:
        result = await db.query("SELECT 1")
        metrics.increment("health_checks")
        return {"status": "healthy", "db": "connected" if result else "disconnected"}
    except Exception as e:
        error_message = f"Health check failed: {str(e)} [server.py:100] [ID:health_error]"
        await security.log_error("unknown", "health", error_message)
        metrics.increment("health_errors")
        raise HTTPException(status_code=503, detail=error_message)

@app.post("/vial2/mcp/api/vials", tags=["vials"])
async def vial_endpoint(data: dict):
    try:
        result = await vial_manager.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "vial", data)
        metrics.increment("vial_requests")
        return result
    except Exception as e:
        error_message = f"Vial operation failed: {str(e)} [server.py:110] [ID:vial_error]"
        await security.log_error(data.get("user_id", "unknown"), "vial", error_message)
        metrics.increment("vial_errors")
        raise HTTPException(status_code=400, detail=error_message)

@app.post("/vial2/mcp/api/wallet", tags=["wallet"])
async def wallet_endpoint(data: dict):
    try:
        result = await wallet_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "wallet", data)
        metrics.increment("wallet_requests")
        return result
    except Exception as e:
        error_message = f"Wallet operation failed: {str(e)} [server.py:120] [ID:wallet_error]"
        await security.log_error(data.get("user_id", "unknown"), "wallet", error_message)
        metrics.increment("wallet_errors")
        raise HTTPException(status_code=400, detail=error_message)

@app.websocket("/vial2/mcp/api/ws", name="WebSocket Notifications")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await notifications.handle_websocket(websocket, "vial2_token")
        metrics.increment("websocket_connections")
    except Exception as e:
        error_message = f"WebSocket failed: {str(e)} [server.py:130] [ID:ws_error]"
        await security.log_error("unknown", "websocket", error_message)
        metrics.increment("websocket_errors")
        raise
