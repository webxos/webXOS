from fastapi import APIRouter, Request, HTTPException
from ..utils.helpers import get_db_pool, log_event
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/api", tags=["webhooks"])

@router.post("/webhooks")
async def handle_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("event_type")
        if not event_type:
            raise ValueError("Missing event_type in webhook payload")
        async with get_db_pool() as db:
            await log_event("webhook", f"Received webhook: {event_type}", db)
        return {"status": "success", "event_type": event_type}
    except Exception as e:
        error_logger.log_error("webhooks", f"Webhook handling failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Webhook handling failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #api #webhooks #neon_mcp
