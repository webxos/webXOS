from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging
from vial.auth_manager import AuthManager
from vial.quantum_simulator import QuantumSimulator
from db.mcp_db_init import DatabaseManager
import traceback

app = FastAPI(title="Vial MCP Backend")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

auth_manager = AuthManager()
db_manager = DatabaseManager()
quantum_simulator = QuantumSimulator()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class PromptRequest(BaseModel):
    vial_id: str
    prompt: str

class TaskRequest(BaseModel):
    vial_id: str
    task: str

class ConfigRequest(BaseModel):
    vial_id: str
    key: str
    value: str

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not api_key or not auth_manager.verify_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

@app.on_event("startup")
async def startup_event():
    try:
        await db_manager.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.disconnect()
    logger.info("Database disconnected")

@app.get("/api/health")
async def health_check():
    try:
        db_status = await db_manager.check_health()
        return {"status": "healthy", "database": db_status}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/auth/login")
async def login(api_key: str = Security(API_KEY_HEADER)):
    try:
        if auth_manager.verify_api_key(api_key):
            token = auth_manager.generate_token(api_key)
            return {"token": token, "message": "Authenticated successfully"}
        raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.post("/api/quantum/link")
async def quantum_link(request: PromptRequest, api_key: str = Depends(verify_api_key)):
    try:
        result = await quantum_simulator.process_quantum_link(request.vial_id, request.prompt)
        await db_manager.store_quantum_state(request.vial_id, result)
        return {"status": "success", "quantum_state": result}
    except Exception as e:
        logger.error(f"Quantum link failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Quantum link failed: {str(e)}")

@app.post("/api/prompt")
async def send_prompt(request: PromptRequest, api_key: str = Depends(verify_api_key)):
    try:
        await db_manager.store_prompt(request.vial_id, request.prompt)
        return {"status": "success", "message": f"Prompt sent to {request.vial_id}"}
    except Exception as e:
        logger.error(f"Prompt processing failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prompt processing failed: {str(e)}")

@app.post("/api/task")
async def assign_task(request: TaskRequest, api_key: str = Depends(verify_api_key)):
    try:
        await db_manager.store_task(request.vial_id, request.task)
        return {"status": "success", "message": f"Task assigned to {request.vial_id}"}
    except Exception as e:
        logger.error(f"Task assignment failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Task assignment failed: {str(e)}")

@app.post("/api/config")
async def set_config(request: ConfigRequest, api_key: str = Depends(verify_api_key)):
    try:
        await db_manager.store_config(request.vial_id, request.key, request.value)
        return {"status": "success", "message": f"Config set for {request.vial_id}"}
    except Exception as e:
        logger.error(f"Config setting failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Config setting failed: {str(e)}")

@app.get("/api/log_error")
async def log_error(error: str, api_key: str = Depends(verify_api_key)):
    try:
        logger.error(f"Client error: {error}")
        await db_manager.store_error(error)
        return {"status": "success", "message": "Error logged"}
    except Exception as e:
        logger.error(f"Error logging failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error logging failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
