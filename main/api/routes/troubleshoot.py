from fastapi import APIRouter, Depends
from utils.auth import verify_token

router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot(token: str = Depends(verify_token)):
    return {
        "status": "success",
        "diagnostics": {
            "cpu_usage_percent": 10,
            "memory_usage_mb": 500,
            "memory_total_mb": 16000,
            "disk_usage_percent": 20,
            "process_memory_mb": 100
        }
    }
