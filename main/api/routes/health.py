from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
from main.api.utils.auth import verify_token

router = APIRouter()

@router.get("/health")
async def health_check(payload: dict = Depends(verify_token)):
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "balance": 38940.0000,
        "reputation": 1200983581,
        "user_id": "a1d57580-d88b-4c90-a0f8-6f2c8511b1e4",
        "address": "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d",
        "vial_agent": "vial1",
        "task_status": "Idle",
        "metrics": {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": memory.used / 1024 / 1024
        }
    }
