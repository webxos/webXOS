from fastapi import APIRouter, Depends
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
import psutil
import os

router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot(user_id: str = Depends(verify_token)):
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        process = psutil.Process(os.getpid())
        diagnostics = {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": memory.used / 1024 / 1024,
            "memory_total_mb": memory.total / 1024 / 1024,
            "disk_usage_percent": disk.percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024
        }
        log_info(f"Troubleshoot diagnostics retrieved for {user_id}")
        return {"status": "success", "diagnostics": diagnostics}
    except Exception as e:
        log_error(f"Troubleshoot failed for {user_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
