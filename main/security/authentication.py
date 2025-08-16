from datetime import datetime, timedelta
from jose import JWTError, jwt
from pymongo import MongoClient
from ..config.settings import settings
from fastapi import HTTPException

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm="HS256")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        return payload
    except JWTError:
        return None

def verify_credentials(api_key: str, api_secret: str):
    client = MongoClient(settings.database.url)
    db = client[settings.database.db_name]
    credentials = db.credentials.find_one({"api_key": api_key, "api_secret": api_secret})
    client.close()
    return credentials
