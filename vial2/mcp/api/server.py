from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from config.config import DatabaseConfig
from tools.auth_tool import AuthTool
from tools.wallet import WalletTool
from tools.vial_management import VialManager
from lib.security import SecurityHandler
from lib.notifications import NotificationHandler
from sql.query_engine import QueryEngine
from postgrest import AsyncPostgrestClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vial MCP API")
db = DatabaseConfig()
auth_tool = AuthTool(db)
wallet_tool = WalletTool(db)
vial_manager = VialManager(db)
security = SecurityHandler(db)
notifications = NotificationHandler()
query_engine = QueryEngine(db)
data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")

@app.on_event("startup")
async def startup():
    await db.connect()
    logger.info(f"Connected to database: {db.url} [server.py:25] [ID:startup_success]")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    await data_api.aclose()
    logger.info("Database and Data API connections closed [server.py:30] [ID:shutdown_success]")

@app.post("/vial2/mcp/api/auth")
async def auth_endpoint(data: dict):
    try:
        result = await auth_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "auth", data)
        return result
    except Exception as e:
        error_message = f"Auth failed: {str(e)} [server.py:35] [ID:auth_error]"
        await security.log_error("unknown", "auth", error_message)
        raise HTTPException(status_code=400, detail=error_message)

@app.post("/vial2/mcp/api/endpoints")
async def command_endpoint(data: dict):
    try:
        user_id = data.get("user_id")
        command = data.get("command")
        project_id = data.get("project_id", "twilight-art-21036984")
        if command == "data":
            table = data.get("args")[0]
            action = data.get("args")[1]
            if table == "sql":
                result = await query_engine.execute_sql(user_id, " ".join(data.get("args")[1:]), data.get("access_token"), project_id)
            else:
                result = await query_engine.execute_data_api_query(user_id, table, action, data.get("access_token"), project_id)
        elif command == "git":
            result = await vial_manager.execute_git(data)
        elif command in ["prompt", "task", "config"]:
            result = await vial_manager.execute(data)
        else:
            error_message = f"Invalid command: {command} [server.py:50] [ID:command_error]"
            raise ValueError(error_message)
        await security.log_action(user_id, command, data)
        return result
    except Exception as e:
        error_message = f"Command failed: {str(e)} [server.py:55] [ID:command_error]"
        await security.log_error(user_id or "unknown", "command", error_message)
        raise HTTPException(status_code=400, detail=error_message)

@app.get("/vial2/mcp/api/health")
async def health_check():
    try:
        result = await db.query("SELECT 1")
        return {"status": "healthy", "db": "connected" if result else "disconnected"}
    except Exception as e:
        error_message = f"Health check failed: {str(e)} [server.py:65] [ID:health_error]"
        await security.log_error("unknown", "health", error_message)
        raise HTTPException(status_code=503, detail=error_message)

@app.post("/vial2/mcp/api/vials")
async def vial_endpoint(data: dict):
    try:
        result = await vial_manager.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "vial", data)
        return result
    except Exception as e:
        error_message = f"Vial operation failed: {str(e)} [server.py:75] [ID:vial_error]"
        await security.log_error(data.get("user_id", "unknown"), "vial", error_message)
        raise HTTPException(status_code=400, detail=error_message)

@app.post("/vial2/mcp/api/wallet")
async def wallet_endpoint(data: dict):
    try:
        result = await wallet_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "wallet", data)
        return result
    except Exception as e:
        error_message = f"Wallet operation failed: {str(e)} [server.py:85] [ID:wallet_error]"
        await security.log_error(data.get("user_id", "unknown"), "wallet", error_message)
        raise HTTPException(status_code=400, detail=error_message)

@app.websocket("/vial2/mcp/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await notifications.handle_websocket(websocket, "vial2_token")
