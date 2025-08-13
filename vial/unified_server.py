from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import pymongo
import redis
import psycopg2
from milvus import Milvus
from weaviate import Client as WeaviateClient
import faiss
import numpy as np
import logging
import datetime
import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from fastapi.responses import JSONResponse, Response
from vial.auth_manager import AuthManager
import subprocess

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connections
mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)
postgres_conn = psycopg2.connect(os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/mcp_db"))
milvus_client = Milvus(host="localhost", port="19530")
weaviate_client = WeaviateClient(os.getenv("WEAVIATE_URI", "http://localhost:8080"))
faiss_index = faiss.IndexFlatL2(768)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class RetrieveRequest(BaseModel):
    user_id: str
    query: str
    source: str
    format: str = "json"  # json or xml
    wallet: Dict[str, Any]

class LLMRequest(BaseModel):
    user_id: str
    prompt: str
    model: str
    format: str = "json"
    wallet: Dict[str, Any]

class GitRequest(BaseModel):
    user_id: str
    command: str  # e.g., "git clone", "git commit"
    repo_url: str
    wallet: Dict[str, Any]

async def rate_limit(request: Request):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    count = redis_client.get(key)
    if count and int(count) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    redis_client.incr(key)
    redis_client.expire(key, 60)

async def transform_response(data: Any, format: str) -> Response:
    try:
        if format == "xml":
            root = ET.Element("response")
            for key, value in data.items():
                child = ET.SubElement(root, key)
                child.text = str(value)
            return Response(content=ET.tostring(root, encoding="unicode"), media_type="application/xml")
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Transformation error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Transformation error: {str(e)}\n")
        raise HTTPException(status_code=500, detail="Data transformation failed")

async def retrieve_data(request: RetrieveRequest, token: str = Depends(oauth2_scheme)) -> Response:
    await rate_limit(request)
    await auth_manager.verify_token(token, ["read:data"])
    try:
        query_embedding = embeddings.embed_query(request.query)
        if request.source == "postgres":
            with postgres_conn.cursor() as cur:
                cur.execute("SELECT data FROM user_data WHERE user_id = %s", (request.user_id,))
                result = cur.fetchone()
                data = {"status": "success", "data": result[0] if result else {}}
        elif request.source == "milvus":
            result = milvus_client.search(collection_name="mcp_vectors", query_vectors=[query_embedding], limit=5)
            data = {"status": "success", "data": result}
        elif request.source == "weaviate":
            result = weaviate_client.query.get("MCPVectors", ["data"]).with_near_vector({"vector": query_embedding}).with_limit(5).do()
            data = {"status": "success", "data": result}
        elif request.source == "pgvector":
            with postgres_conn.cursor() as cur:
                cur.execute("SELECT data FROM vectors WHERE vector <-> %s::vector LIMIT 5", (query_embedding,))
                result = cur.fetchall()
                data = {"status": "success", "data": [r[0] for r in result]}
        elif request.source == "faiss":
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = faiss_index.search(query_vector, 5)
            data = {"status": "success", "data": {"distances": distances.tolist(), "indices": indices.tolist()}}
        else:
            raise ValueError("Invalid data source")
        
        # Update wallet
        wallet = request.wallet
        wallet["transactions"].append({
            "type": "retrieve",
            "source": request.source,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
        db.collection("wallet").update_one(
            {"user_id": request.user_id},
            {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
            upsert=True
        )
        
        return await transform_response(data, request.format)
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Retrieval error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

async def call_llm(request: LLMRequest, token: str = Depends(oauth2_scheme)) -> Response:
    await rate_limit(request)
    await auth_manager.verify_token(token, ["read:llm"])
    try:
        headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"}
        payload = {"prompt": request.prompt, "model": request.model}
        response = requests.post("https://api.huggingface.co/v1/inference", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Update wallet
        wallet = request.wallet
        wallet["transactions"].append({
            "type": "llm_call",
            "model": request.model,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
        db.collection("wallet").update_one(
            {"user_id": request.user_id},
            {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
            upsert=True
        )
        
        return await transform_response({"status": "success", "response": result}, request.format)
    except Exception as e:
        logger.error(f"LLM call error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** LLM call error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_git_command(request: GitRequest, token: str = Depends(oauth2_scheme)) -> Response:
    await rate_limit(request)
    await auth_manager.verify_token(token, ["write:git"])
    try:
        # Sanitize command
        allowed_commands = ["git clone", "git commit", "git push", "git pull"]
        if not any(request.command.startswith(cmd) for cmd in allowed_commands):
            raise ValueError("Invalid Git command")
        
        # Execute Git command
        result = subprocess.run(request.command, shell=True, capture_output=True, text=True, cwd="/tmp/repo")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        
        # Update wallet
        wallet = request.wallet
        wallet["transactions"].append({
            "type": "git_command",
            "command": request.command,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
        db.collection("wallet").update_one(
            {"user_id": request.user_id},
            {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
            upsert=True
        )
        
        return await transform_response({"status": "success", "output": result.stdout}, "json")
    except Exception as e:
        logger.error(f"Git command error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Git command error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/api/retrieve")
async def api_retrieve(request: RetrieveRequest, token: str = Depends(oauth2_scheme)):
    return await retrieve_data(request, token)

@app.post("/v1/api/llm")
async def api_llm(request: LLMRequest, token: str = Depends(oauth2_scheme)):
    return await call_llm(request, token)

@app.post("/v1/api/git")
async def api_git(request: GitRequest, token: str = Depends(oauth2_scheme)):
    return await execute_git_command(request, token)
