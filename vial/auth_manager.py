import jwt
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
import requests
import os
from typing import List
import datetime

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "VIAL_MCP_SECRET_2025")
        self.oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
        self.oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        self.oauth_token_url = "https://github.com/login/oauth/access_token"
        self.algorithm = "HS256"

    def create_token(self, user_id: str, roles: List[str]) -> str:
        payload = {
            "user_id": user_id,
            "roles": roles,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    async def verify_token(self, token: str, required_roles: List[str]) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_roles = payload.get("roles", [])
            if not all(role in user_roles for role in required_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return payload
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    async def verify_oauth_token(self, code: str) -> dict:
        try:
            response = requests.post(
                self.oauth_token_url,
                data={
                    "client_id": self.oauth_client_id,
                    "client_secret": self.oauth_client_secret,
                    "code": code
                },
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            token_data = response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=401, detail="OAuth token retrieval failed")
            
            # Fetch user info
            user_response = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            
            # Assign roles based on open-source project contribution
            roles = ["read:data", "read:llm"]  # Default roles
            if user_data.get("login") in ["contributor1", "contributor2"]:  # Example contributors
                roles.append("write:git")
            
            return {
                "user_id": user_data.get("login"),
                "roles": roles,
                "access_token": access_token
            }
        except Exception as e:
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** OAuth error: {str(e)}\n")
            raise HTTPException(status_code=401, detail=str(e))

    async def generate_api_key(self, user_id: str) -> str:
        try:
            api_key = self.create_token(user_id, ["read:data", "read:llm"])
            with psycopg2.connect(os.getenv("POSTGRES_URI")) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO api_keys (user_id, api_key) VALUES (%s, %s)", (user_id, api_key))
                    conn.commit()
            return api_key
        except Exception as e:
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** API key generation error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))
