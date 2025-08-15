# main/server/mcp/events/pubsub_manager.py
import redis
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
import os
import json
from datetime import datetime

app = FastAPI(title="Vial MCP PubSub Manager")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()

class PubSubEvent(BaseModel):
    channel: str
    data: Dict
    user_id: str

class PubSubManager:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )

    def publish(self, channel: str, data: Dict) -> bool:
        with self.metrics.track_span("publish_event", {"channel": channel}):
            try:
                self.redis_client.publish(channel, json.dumps(data))
                return True
            except Exception as e:
                handle_generic_error(e, context="publish_event")
                return False

    def subscribe(self, channel: str):
        with self.metrics.track_span("subscribe_channel", {"channel": channel}):
            try:
                pubsub = self.redis_client.pubsub()
                pubsub.subscribe(channel)
                return pubsub
            except Exception as e:
                handle_generic_error(e, context="subscribe_channel")
                raise

pubsub_manager = PubSubManager()

@app.post("/events/publish")
async def publish_event(event: PubSubEvent, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("publish_event_endpoint", {"channel": event.channel, "user_id": event.user_id}):
        try:
            metrics.verify_token(token)
            event_data = event.data
            event_data["timestamp"] = datetime.utcnow().isoformat()
            event_data["user_id"] = event.user_id
            success = pubsub_manager.publish(event.channel, event_data)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to publish event")
            return {"status": "success", "message": f"Event published to {event.channel}"}
        except Exception as e:
            handle_generic_error(e, context="publish_event_endpoint")
            raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}")
