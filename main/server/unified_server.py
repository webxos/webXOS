from fastapi import FastAPI,HTTPException,Depends,Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from pydantic import BaseModel
import uvicorn,logging,traceback,sqlite3,os,json,asyncio,redis.asyncio as redis
from datetime import datetime
from vial.auth_manager import AuthManager
from vial.quantum_simulator import QuantumSimulator
from vial.agent1 import NomicAgent
from vial.agent2 import CogniTALLMwareAgent
from vial.agent3 import LLMwareAgent
from vial.agent4 import JinaAIAgent

app=FastAPI(title="Vial MCP Backend",description="Secure backend for Vial MCP with wallet-based authentication and SQLite storage")
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)
app.add_middleware(CORSMiddleware,allow_origins=["https://localhost:8000"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

auth_manager=AuthManager()
quantum_simulator=QuantumSimulator()
nomic_agent=NomicAgent()
cogni_agent=CogniTALLMwareAgent()
llmware_agent=LLMwareAgent()
jina_agent=JinaAIAgent()
NOTES_DIR="/app/notes"
os.makedirs(NOTES_DIR,exist_ok=True)
API_KEY_HEADER=APIKeyHeader(name="X-API-Key",auto_error=False)

class PromptRequest(BaseModel):
    vial_id:str
    prompt:str
    wallet_id:str

class TaskRequest(BaseModel):
    vial_id:str
    task:str
    wallet_id:str

class ConfigRequest(BaseModel):
    vial_id:str
    key:str
    value:str
    wallet_id:str

class NoteRequest(BaseModel):
    wallet_id:str
    content:str
    resource_id:str|None=None

class ResourceRequest(BaseModel):
    wallet_id:str
    resource_id:str|None=None

async def verify_api_key(api_key:str=Security(API_KEY_HEADER)):
    """Verify API key and wallet association.

    Args:
        api_key (str): API key from request header.

    Returns:
        str: Verified API key.

    Raises:
        HTTPException: If API key is invalid or missing.
    """
    if not api_key or not auth_manager.verify_api_key(api_key):
        logger.error("Invalid or missing API key")
        raise HTTPException(status_code=401,detail="Invalid or missing API key")
    return api_key

async def init_redis():
    """Initialize Redis for rate limiting."""
    try:
        redis_client=redis.Redis(host='redis',port=6379,decode_responses=True)
        await FastAPILimiter.init(redis_client)
        logger.info("Redis initialized for rate limiting")
    except Exception as e:
        logger.error(f"Redis initialization failed: {str(e)}")
        raise HTTPException(status_code=500,detail=f"Redis initialization failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize database and Redis on startup."""
    try:
        await init_redis()
        init_db()
        logger.info("Backend started with SQLite and Redis")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database and Redis connections on shutdown."""
    try:
        close_db()
        await FastAPILimiter.close()
        logger.info("Backend shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown failed: {str(e)}\n{traceback.format_exc()}")

def init_db():
    """Initialize SQLite database with tables for notes, prompts, tasks, configs, and quantum states."""
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT,wallet_id TEXT NOT NULL,content TEXT NOT NULL,resource_id TEXT,timestamp TEXT NOT NULL)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS prompts (id INTEGER PRIMARY KEY AUTOINCREMENT,vial_id TEXT NOT NULL,prompt TEXT NOT NULL,wallet_id TEXT NOT NULL,timestamp TEXT NOT NULL)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT,vial_id TEXT NOT NULL,task TEXT NOT NULL,wallet_id TEXT NOT NULL,timestamp TEXT NOT NULL)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS configs (id INTEGER PRIMARY KEY AUTOINCREMENT,vial_id TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,wallet_id TEXT NOT NULL,timestamp TEXT NOT NULL)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS quantum_states (id INTEGER PRIMARY KEY AUTOINCREMENT,vial_id TEXT NOT NULL,state TEXT NOT NULL,wallet_id TEXT NOT NULL,timestamp TEXT NOT NULL)""")
            conn.commit()
            logger.info("SQLite database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"Database initialization failed: {str(e)}")

def close_db():
    """Close SQLite database connection."""
    try:
        logger.info("SQLite database closed")
    except Exception as e:
        logger.error(f"Database close failed: {str(e)}\n{traceback.format_exc()}")

@app.get("/health",response_class=JSONResponse)
async def health_check():
    """Check backend and database health.

    Returns:
        dict: Health status of backend and database.

    Raises:
        HTTPException: If health check fails.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            conn.execute("SELECT 1")
        return {"status":"healthy","database":"healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Health check failed: {str(e)}")

@app.post("/api/prompt")
async def send_prompt(request:PromptRequest,api_key:str=Depends(verify_api_key)):
    """Store a prompt for a vial with wallet verification.

    Args:
        request (PromptRequest): Prompt data with vial_id, prompt, and wallet_id.
        api_key (str): Verified API key.

    Returns:
        dict: Success message.

    Raises:
        HTTPException: If prompt storage fails.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("INSERT INTO prompts (vial_id,prompt,wallet_id,timestamp) VALUES (?,?,?,?)",(request.vial_id,request.prompt,request.wallet_id,datetime.now().isoformat()))
            conn.commit()
        logger.info(f"Stored prompt for vial {request.vial_id} by wallet {request.wallet_id}")
        return {"status":"success","message":f"Prompt sent to {request.vial_id}"}
    except Exception as e:
        logger.error(f"Prompt storage failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Prompt storage failed: {str(e)}")

@app.post("/api/task")
async def assign_task(request:TaskRequest,api_key:str=Depends(verify_api_key)):
    """Assign a task to a vial with wallet verification.

    Args:
        request (TaskRequest): Task data with vial_id, task, and wallet_id.
        api_key (str): Verified API key.

    Returns:
        dict: Success message.

    Raises:
        HTTPException: If task storage fails.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("INSERT INTO tasks (vial_id,task,wallet_id,timestamp) VALUES (?,?,?,?)",(request.vial_id,request.task,request.wallet_id,datetime.now().isoformat()))
            conn.commit()
        logger.info(f"Stored task for vial {request.vial_id} by wallet {request.wallet_id}")
        return {"status":"success","message":f"Task assigned to {request.vial_id}"}
    except Exception as e:
        logger.error(f"Task storage failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Task storage failed: {str(e)}")

@app.post("/api/config")
async def set_config(request:ConfigRequest,api_key:str=Depends(verify_api_key)):
    """Set configuration for a vial with wallet verification.

    Args:
        request (ConfigRequest): Config data with vial_id, key, value, and wallet_id.
        api_key (str): Verified API key.

    Returns:
        dict: Success message.

    Raises:
        HTTPException: If config storage fails.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO configs (vial_id,key,value,wallet_id,timestamp) VALUES (?,?,?,?,?)",(request.vial_id,request.key,request.value,request.wallet_id,datetime.now().isoformat()))
            conn.commit()
        logger.info(f"Stored config for vial {request.vial_id}: {request.key}={request.value} by wallet {request.wallet_id}")
        return {"status":"success","message":f"Config set for {request.vial_id}"}
    except Exception as e:
        logger.error(f"Config storage failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Config storage failed: {str(e)}")

@app.post("/api/quantum/link")
async def quantum_link(request:PromptRequest,api_key:str=Depends(verify_api_key)):
    """Process quantum link for a vial with wallet verification.

    Args:
        request (PromptRequest): Prompt data with vial_id, prompt, and wallet_id.
        api_key (str): Verified API key.

    Returns:
        dict: Quantum state result.

    Raises:
        HTTPException: If quantum link processing fails.
    """
    try:
        result=await quantum_simulator.process_quantum_link(request.vial_id,request.prompt)
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO quantum_states (vial_id,state,wallet_id,timestamp) VALUES (?,?,?,?)",(request.vial_id,json.dumps(result),request.wallet_id,datetime.now().isoformat()))
            conn.commit()
        logger.info(f"Processed quantum link for vial {request.vial_id} by wallet {request.wallet_id}")
        return {"status":"success","quantum_state":result}
    except Exception as e:
        logger.error(f"Quantum link failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Quantum link failed: {str(e)}")

@app.post("/api/notes/add")
async def add_note(request:NoteRequest,api_key:str=Depends(verify_api_key)):
    """Add a note to the SQLite database and save to file system.

    Args:
        request (NoteRequest): Note data with wallet_id, content, and optional resource_id.
        api_key (str): Verified API key.

    Returns:
        dict: Success message with note ID.

    Raises:
        HTTPException: If note storage fails.
    """
    try:
        note_id=None
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("INSERT INTO notes (wallet_id,content,resource_id,timestamp) VALUES (?,?,?,?)",(request.wallet_id,request.content,request.resource_id,datetime.now().isoformat()))
            note_id=cursor.lastrowid
            conn.commit()
        note_path=os.path.join(NOTES_DIR,f"note_{note_id}_{request.wallet_id}.txt")
        with open(note_path,"w") as f:
            f.write(request.content)
        logger.info(f"Stored note {note_id} for wallet {request.wallet_id}")
        return {"status":"success","message":"Note added","note_id":note_id}
    except Exception as e:
        logger.error(f"Note storage failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Note storage failed: {str(e)}")

@app.get("/api/notes/read/{note_id}")
async def read_note(note_id:int,wallet_id:str,api_key:str=Depends(verify_api_key)):
    """Read a note from SQLite by ID and wallet verification.

    Args:
        note_id (int): ID of the note to read.
        wallet_id (str): Wallet ID for verification.
        api_key (str): Verified API key.

    Returns:
        dict: Note content and metadata.

    Raises:
        HTTPException: If note retrieval fails or access is unauthorized.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE id=? AND wallet_id=?",(note_id,wallet_id))
            note=cursor.fetchone()
            if not note:
                logger.warning(f"Note {note_id} not found or unauthorized for wallet {wallet_id}")
                raise HTTPException(status_code=404,detail="Note not found or unauthorized")
            return {"status":"success","note":{"id":note[0],"content":note[1],"resource_id":note[2],"timestamp":note[3]}}
    except Exception as e:
        logger.error(f"Note read failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Note read failed: {str(e)}")

@app.get("/api/resources/latest")
async def get_latest_resources(wallet_id:str,api_key:str=Depends(verify_api_key)):
    """Get latest notes as resources for a wallet.

    Args:
        wallet_id (str): Wallet ID for verification.
        api_key (str): Verified API key.

    Returns:
        dict: List of latest notes.

    Raises:
        HTTPException: If resource retrieval fails.
    """
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor=conn.cursor()
            cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE wallet_id=? ORDER BY timestamp DESC LIMIT 10",(wallet_id,))
            notes=cursor.fetchall()
            resources=[{"id":n[0],"content":n[1],"resource_id":n[2],"timestamp":n[3]} for n in notes]
            logger.info(f"Retrieved {len(resources)} resources for wallet {wallet_id}")
            return {"status":"success","resources":resources}
    except Exception as e:
        logger.error(f"Resource retrieval failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,detail=f"Resource retrieval failed: {str(e)}")

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000,ssl_keyfile="/app/certs/key.pem",ssl_certfile="/app/certs/cert.pem")
