from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import aiohttp
import logging
from config.config import DatabaseConfig
from tools.auth_tool import AuthTool
from tools.api_key_manager import ApiKeyManager
from tools.wallet_validator import WalletValidator
from tools.git_tool import GitTool
from tools.vial_management import VialManagement
from tools.replication_manager import ReplicationManager
from tools.relay_signal import RelaySignal
from tools.compute_ctl import ComputeCtl
from tools.compute_monitor import ComputeMonitor
from lib.security import SecurityHandler
import json

app = FastAPI()
security = HTTPBearer()
logger = logging.getLogger(__name__)

async def get_db():
    db = DatabaseConfig()
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)} [server.py:25] [ID:token_error]")
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/vial2/mcp/api/endpoints")
async def endpoints(data: dict, user_id: str = Depends(verify_token), db: DatabaseConfig = Depends(get_db)):
    try:
        method = data.get("method")
        command = data.get("command")
        project_id = data.get("project_id", db.project_id)
        if project_id != db.project_id:
            error_message = f"Invalid project ID: {project_id} [server.py:30] [ID:project_error]"
            logger.error(error_message)
            return {"error": error_message}
        
        if method == "authenticate":
            auth_tool = AuthTool(db)
            async with aiohttp.ClientSession() as session:
                result = await auth_tool.authenticate(data.get("code"), data.get("redirect_uri"), session)
                if result.get("error"):
                    logger.error(f"Authentication failed: {result['error']} [server.py:35] [ID:auth_error]")
                    return result
                return result
        elif method == "generate_api_key":
            api_key_manager = ApiKeyManager(db)
            return await api_key_manager.generate_api_key(user_id, project_id)
        elif method == "validate_md_wallet":
            wallet_validator = WalletValidator(db)
            return await wallet_validator.validate_wallet(user_id, data.get("wallet_data"), project_id)
        elif method == "configure":
            compute_ctl = ComputeCtl(db)
            return await compute_ctl.initialize_compute(user_id, data.get("spec", {}))
        elif method == "refresh_configuration":
            compute_ctl = ComputeCtl(db)
            return await compute_ctl.initialize_compute(user_id, data.get("spec", {}))
        elif method == "terminate_fast":
            return await db.query(
                "UPDATE computes SET state = $1 WHERE project_id = $2",
                ["TerminationPendingFast", project_id]
            )
        elif method == "terminate_immediate":
            return await db.query(
                "UPDATE computes SET state = $1 WHERE project_id = $2",
                ["TerminationPendingImmediate", project_id]
            )
        elif command and command.startswith("/git"):
            git_tool = GitTool(db)
            return await git_tool.execute(command, user_id, project_id)
        elif command and command.startswith(("/prompt", "/task", "/config", "/status")):
            vial_management = VialManagement(db)
            return await vial_management.execute(command, user_id, project_id)
        elif command and command.startswith("/replication"):
