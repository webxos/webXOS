from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
import logging
import datetime
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthRequest(BaseModel):
    user_id: str
    password: str

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "VIAL_MCP_SECRET_2025")

    async def authenticate(self, user_id: str, password: str) -> dict:
        try:
            # Mock authentication (replace with actual user DB check in production)
            if not user_id or not password:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            token = jwt.encode({
                "user_id": user_id,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, self.secret_key, algorithm="HS256")
            
            db = pymongo.MongoClient("mongodb://localhost:27017")["mcp_db"]
            db.collection("auth_logs").insert_one({
                "user_id": user_id,
                "action": "login",
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            
            return {"token": token}
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Authentication error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

auth_manager = AuthManager()

@app.post("/api/authenticate")
async def authenticate_user(request: AuthRequest):
    return await auth_manager.authenticate(request.user_id, request.password)
