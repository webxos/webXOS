from fastapi import APIRouter, HTTPException
from monitoring.metrics import MetricsCollector
from config.config import DatabaseConfig
from lib.security import SecurityHandler
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vial2/mcp/api/monitoring", tags=["monitoring"])

@router.get("/metrics")
async def get_metrics():
    try:
        metrics = MetricsCollector()
        result = await metrics.get_metrics()
        if "error" in result:
            error_message = f"Metrics retrieval failed: {result['error']} [endpoints.py:15] [ID:metrics_error]"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
        logger.info("Metrics retrieved [endpoints.py:20] [ID:metrics_success]")
        return result
    except Exception as e:
        error_message = f"Metrics endpoint failed: {str(e)} [endpoints.py:25] [ID:metrics_endpoint_error]"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/replication_status")
async def replication_status():
    try:
        db = DatabaseConfig()
        security = SecurityHandler(db)
        result = await db.query("SELECT subname, received_lsn, latest_end_lsn, last_msg_receipt_time FROM pg_stat_subscription")
        await security.log_action("system", "replication_status", {"subscriptions": [r["subname"] for r in result]})
        logger.info("Replication status retrieved [endpoints.py:30] [ID:replication_status_success]")
        return {"status": "success", "subscriptions": result}
    except Exception as e:
        error_message = f"Replication status failed: {str(e)} [endpoints.py:35] [ID:replication_status_error]"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
