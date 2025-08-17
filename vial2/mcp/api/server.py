from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from config.config import DatabaseConfig
from tools.auth_tool import AuthTool
from tools.wallet import WalletTool
from tools.vial_management import VialManager
from lib.security import SecurityHandler
from lib.notifications import NotificationHandler
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
data_api = AsyncPostgrestClient("https://app-billowing-king-08029676.dpl.myneon.app")

@app.on_event("startup")
async def startup():
    await db.connect()
    logger.info(f"Connected to database: {db.url}")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    await data_api.aclose()
    logger.info("Database and Data API connections closed")

@app.post("/vial2/mcp/api/auth")
async def auth_endpoint(data: dict):
    try:
        result = await auth_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "auth", data)
        return result
    except Exception as e:
        await security.log_error("unknown", "auth", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial2/mcp/api/endpoints")
async def command_endpoint(data: dict):
    try:
        user_id = data.get("user_id")
        command = data.get("command")
        project_id = data.get("project_id", "twilight-art-21036984")
        if command == "data":
            table = data.get("args")[0]
            action = data.get("args")[1]
            result = await data_api_query(user_id, table, action, data.get("access_token"), project_id)
        elif command in ["prompt", "task", "config"]:
            result = await vial_manager.execute(data)
        elif command == "git":
            result = await vial_manager.execute_git(data)
        else:
            raise ValueError("Invalid command")
        await security.log_action(user_id, command, data)
        return result
    except Exception as e:
        await security.log_error(user_id or "unknown", "command", str(e))
        raise HTTPException(status_code=400, detail=str(e))

async def data_api_query(user_id: str, table: str, action: str, token: str, project_id: str):
    try:
        data_api.auth(token)
        if action == "select":
            response = await data_api.from_(table).select("*").eq("user_id", user_id).eq("project_id", project_id).execute()
        elif action == "insert":
            response = await data_api.from_(table).insert({"user_id": user_id, "project_id": project_id}).execute()
        elif action == "update":
            response = await data_api.from_(table).update({"user_id": user_id}).eq("user_id", user_id).eq("project_id", project_id).execute()
        elif action == "delete":
            response = await data_api.from_(table).delete().eq("user_id", user_id).eq("project_id", project_id).execute()
        else:
            raise ValueError("Invalid Data API action")
        return response.data
    except Exception as e:
        raise ValueError(f"Data API query failed: {str(e)}")

@app.get("/vial2/mcp/api/health")
async def health_check():
    try:
        result = await db.query("SELECT 1")
        return {"status": "healthy", "db": "connected" if result else "disconnected"}
    except Exception as e:
        await security.log_error("unknown", "health", str(e))
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/vial2/mcp/api/vials")
async def vial_endpoint(data: dict):
    try:
        result = await vial_manager.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "vial", data)
        return result
    except Exception as e:
        await security.log_error(data.get("user_id", "unknown"), "vial", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vial2/mcp/api/wallet")
async def wallet_endpoint(data: dict):
    try:
        result = await wallet_tool.execute(data)
        await security.log_action(data.get("user_id", "unknown"), "wallet", data)
        return result
    except Exception as e:
        await security.log_error(data.get("user_id", "unknown"), "wallet", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/vial2/mcp/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await notifications.handle_websocket(websocket, "vial2_token")
