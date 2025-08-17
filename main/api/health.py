from fastapi import APIRouter
from ...utils.logging import log_info
from ...utils.monitoring import monitor

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        metrics = await monitor.collect_metrics()
        log_info("Health check passed")
        return {"status": "healthy", "metrics": metrics}
    except Exception:
        log_info("Health check failed")
        return {"status": "unhealthy"}
