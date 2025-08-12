from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import os
import json
from queen_agent import QueenAgent
import logging
from logging_config import setup_logging
from security import validate_credentials, sanitize_input
import secrets

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
queen = QueenAgent()

class AuthRequest(BaseModel):
    username: str
    password: str

class PromptRequest(BaseModel):
    api_key: str
    prompt: str

@app.get("/")
async def root():
    return {"message": "Welcome to Hive MCP Server. Access /hive.html for the interface."}

@app.post("/hive/generate_credentials")
async def generate_credentials():
    try:
        username = f"user_{str(uuid.uuid4())[:8]}"
        password = secrets.token_urlsafe(12)
        api_key = str(uuid.uuid4())
        os.makedirs(os.path.dirname("users.txt"), exist_ok=True)
        with open("users.txt", "a") as f:
            f.write(f"{username}:{password}:{api_key}\n")
        logger.info(f"Generated credentials for {username}")
        return {"username": username, "password": password}
    except Exception as e:
        logger.error(f"Credential generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/hive/authenticate")
async def authenticate(request: AuthRequest):
    try:
        sanitized_username = sanitize_input(request.username)
        sanitized_password = sanitize_input(request.password)
        if not os.path.exists("users.txt"):
            logger.error("users.txt not found")
            raise HTTPException(status_code=500, detail="Authentication service unavailable")
        with open("users.txt", "r") as f:
            for line in f:
                stored_username, stored_password, api_key = line.strip().split(":")
                if stored_username == sanitized_username and stored_password == sanitized_password:
                    logger.info(f"Authenticated {sanitized_username}")
                    return {"api_key": api_key}
        logger.error(f"Authentication failed for {sanitized_username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/hive/submit_prompt")
async def submit_prompt(request: PromptRequest):
    try:
        request_id = str(uuid.uuid4())
        sanitized_prompt = sanitize_input(request.prompt)
        success = await queen.spawn_agent(request_id, sanitized_prompt, request.api_key)
        if not success:
            logger.error(f"Prompt submission failed for request {request_id}: Invalid API key or max agents reached")
            raise HTTPException(status_code=403, detail="Invalid API key or max agents reached")
        logger.info(f"Prompt submitted for request {request_id}")
        return {"response": f"Request {request_id} is being processed"}
    except Exception as e:
        logger.error(f"Prompt submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/hive/logs")
async def get_logs():
    try:
        logs = []
        if not os.path.exists("logs"):
            logger.info("No logs found, returning empty list")
            return {"logs": []}
        for log_file in os.listdir("logs"):
            try:
                with open(f"logs/{log_file}", "r") as f:
                    logs.append(json.load(f))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in log file {log_file}: {str(e)}")
                continue
        logger.info("Logs retrieved successfully")
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Log retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")