# main/server/mcp/utils/webhook_manager.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
import requests
from datetime import datetime
import os

app = FastAPI(title="Vial MCP Webhook Manager")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class Webhook(BaseModel):
    user_id: str
    url: str
    event_type: str

class WebhookResponse(BaseModel):
    webhook_id: str
    user_id: str
    url: str
    event_type: str
    timestamp: str

@app.post("/webhooks", response_model=WebhookResponse)
async def register_webhook(webhook: Webhook, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("register_webhook", {"user_id": webhook.user_id, "event_type": webhook.event_type}):
        try:
            metrics.verify_token(token)
            webhook_data = webhook.dict()
            webhook_data["timestamp"] = datetime.utcnow()
            webhook_id = db_manager.insert_one("webhooks", webhook_data)
            return WebhookResponse(webhook_id=webhook_id, **webhook_data)
        except Exception as e:
            handle_generic_error(e, context="register_webhook")
            raise HTTPException(status_code=500, detail=f"Failed to register webhook: {str(e)}")

@app.post("/webhooks/notify/{event_type}")
async def notify_webhooks(event_type: str, event_data: Dict, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("notify_webhooks", {"event_type": event_type}):
        try:
            metrics.verify_token(token)
            webhooks = db_manager.find_many("webhooks", {"event_type": event_type})
            for webhook in webhooks:
                try:
                    requests.post(webhook["url"], json=event_data, timeout=5)
                    db_manager.insert_one("webhook_logs", {
                        "webhook_id": webhook["_id"],
                        "event_type": event_type,
                        "timestamp": datetime.utcnow(),
                        "status": "success"
                    })
                except requests.RequestException as e:
                    db_manager.insert_one("webhook_logs", {
                        "webhook_id": webhook["_id"],
                        "event_type": event_type,
                        "timestamp": datetime.utcnow(),
                        "status": "failed",
                        "error": str(e)
                    })
                    metrics.record_error("webhook_notify_failure", f"Failed to notify {webhook['url']}")
            return {"status": "success", "message": f"Notified {len(webhooks)} webhooks"}
        except Exception as e:
            handle_generic_error(e, context="notify_webhooks")
            raise HTTPException(status_code=500, detail=f"Failed to notify webhooks: {str(e)}")
