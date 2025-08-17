from fastapi import APIRouter
from ...utils.logging import log_error, log_info
import torch
import tensorflow as tf
import dspy
import platform
import psutil

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        # Enhanced health check with system and ML library diagnostics
        system_info = {
            "status": "ok",
            "balance": 0.0000,
            "reputation": 0,
            "system": platform.system(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "pytorch_version": torch.__version__,
            "tensorflow_version": tf.__version__,
            "dspy_config": str(dspy.settings.lm)
        }
        log_info("Health check successful with diagnostics")
        return system_info
    except Exception as e:
        log_error(f"Traceback: Health check failed: {str(e)}")
        return {"status": "error", "detail": str(e)}, 500
