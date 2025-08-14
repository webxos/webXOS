# server/mcp/auth/mcp_auth_server.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from .auth_manager import AuthManager

app = FastAPI(title="Vial MCP Auth Server")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
auth_manager = AuthManager()

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    password: str  # Note: In production, use proper password hashing

@app.post("/token", response_model=Token)
async def login_for_access_token(user: User):
    # Simplified authentication (replace with proper user validation)
    if user.username != "admin" or user.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_manager.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    payload = auth_manager.verify_token(token)
    return {"username": payload["sub"]}
