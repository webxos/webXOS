import jwt
import httpx
from fastapi import HTTPException
from .config import config
import logging

logger = logging.getLogger(__name__)

async def handle_auth(method: str, request: dict, db):
    if method == "authenticate":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.stack-auth.com/api/v1/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "client_id": config.STACK_AUTH_CLIENT_ID,
                        "client_secret": config.STACK_AUTH_SECRET_KEY,
                        "code": request.get("code"),
                        "redirect_uri": request.get("redirect_uri")
                    }
                )
                response.raise_for_status()
                data = response.json()
                access_token = data.get("access_token")
                user_info = jwt.decode(access_token, options={"verify_signature": False})
                
                async with db:
                    await db.execute(
                        "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
                        f"0x{user_info.get('sub', 'default')[:40]}"
                    )
                return {"access_token": access_token, "wallet_address": f"0x{user_info.get('sub', 'default')[:40]}"}
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail="Invalid auth method")

async def generate_api_key(user_id: str, db):
    try:
        api_key = jwt.encode({"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=30)}, config.JWT_SECRET_KEY, algorithm="HS256")
        api_secret = jwt.encode({"user_id": user_id, "scope": "api"}, config.JWT_SECRET_KEY, algorithm="HS256")
        async with db:
            await db.execute(
                "UPDATE users SET api_key=$1 WHERE wallet_address=$2",
                api_key, user_id
            )
        return {"api_key": api_key, "api_password": api_secret}
    except Exception as e:
        logger.error(f"API key generation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #auth #stack_auth #neon_mcp
