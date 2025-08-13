import hashlib
import datetime
import jwt
import os
from fastapi import HTTPException

class AuthManager:
    def __init__(self):
        self.secret = os.getenv("JWT_SECRET", "supersecretkey")

    def generate_token(self, user_id: str) -> str:
        try:
            payload = {
                "userId": user_id,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
                "iat": datetime.datetime.utcnow()
            }
            token = jwt.encode(payload, self.secret, algorithm="HS256")
            return token
        except Exception as e:
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Token generation error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Expired token error\n")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Invalid token error\n")
            raise HTTPException(status_code=401, detail="Invalid token")
