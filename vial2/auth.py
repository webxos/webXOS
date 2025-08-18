import jwt
import httpx
from fastapi import HTTPException
from .config import DATABASE_URL, STACK_AUTH_CLIENT_ID, STACK_AUTH_CLIENT_SECRET, JWT_SECRET_KEY
import logging
import asyncpg

logger = logging.getLogger(__name__)

async def handle_auth(method: str, request: dict):
    db = await asyncpg.connect(DATABASE_URL)
    try:
        if method == "authenticate":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.stack-auth.com/api/v1/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "client_id": STACK_AUTH_CLIENT_ID,
                        "client_secret": STACK_AUTH_CLIENT_SECRET,
                        "code": request.get("code"),
                        "redirect_uri": request.get("redirect_uri")
                    }
                )
                response.raise_for_status()
                data = response.json()
                access_token = data.get("access_token")
                user_info = jwt.decode(access_token, options={"verify_signature": False})
                
                await db.execute(
                    "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
                    f"0x{user_info.get('sub', 'default')[:40]}"
                )
                return {"access_token": access_token, "wallet_address": f"0x{user_info.get('sub', 'default')[:40]}"}
    finally:
        await db.close()
    raise HTTPException(status_code=400, detail="Invalid auth method")

async def generate_api_key(user_id: str):
    db = await asyncpg.connect(DATABASE_URL)
    try:
        api_key = jwt.encode({"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=30)}, JWT_SECRET_KEY, algorithm="HS256")
        api_secret = jwt.encode({"user_id": user_id, "scope": "api"}, JWT_SECRET_KEY, algorithm="HS256")
        await db.execute(
            "UPDATE users SET api_key=$1 WHERE wallet_address=$2",
            api_key, user_id
        )
        return {"api_key": api_key, "api_password": api_secret}
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #auth #stack_auth #neon_mcp
