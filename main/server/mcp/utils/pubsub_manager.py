# main/server/mcp/utils/pubsub_manager.py
from typing import Dict, Any, List
import redis.asyncio as redis
from fastapi import APIRouter
from ..utils.mcp_error_handler import MCPError
import os
import logging
import json
import asyncio

router = APIRouter()
logger = logging.getLogger("mcp")

class PubSubManager:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URI", "redis://localhost:6379"))
        self.subscriptions = {}  # In-memory for simplicity; persist in production

    async def subscribe(self, user_id: str, channel: str, event_types: List[str]) -> str:
        try:
            if not user_id or not channel or not event_types:
                raise MCPError(code=-32602, message="User ID, channel, and event types are required")
            subscription_id = secrets.token_hex(16)
            self.subscriptions[subscription_id] = {
                "user_id": user_id,
                "channel": channel,
                "event_types": event_types
            }
            await self.redis_client.set(f"subscription:{subscription_id}", json.dumps(self.subscriptions[subscription_id]))
            logger.info(f"User {user_id} subscribed to channel {channel}")
            return subscription_id
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Subscription failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to subscribe: {str(e)}")

    async def publish(self, channel: str, event_type: str, data: Dict[str, Any]) -> None:
        try:
            if not channel or not event_type:
                raise MCPError(code=-32602, message="Channel and event type are required")
            message = json.dumps({"event_type": event_type, "data": data})
            await self.redis_client.publish(f"channel:{channel}", message)
            logger.info(f"Published event {event_type} to channel {channel}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Publish failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to publish: {str(e)}")

    async def unsubscribe(self, subscription_id: str, user_id: str) -> None:
        try:
            if subscription_id not in self.subscriptions or self.subscriptions[subscription_id]["user_id"] != user_id:
                raise MCPError(code=-32003, message="Subscription not found or access denied")
            del self.subscriptions[subscription_id]
            await self.redis_client.delete(f"subscription:{subscription_id}")
            logger.info(f"User {user_id} unsubscribed from subscription {subscription_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Unsubscribe failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to unsubscribe: {str(e)}")

    async def listen(self, subscription_id: str, user_id: str) -> async_generator:
        try:
            if subscription_id not in self.subscriptions or self.subscriptions[subscription_id]["user_id"] != user_id:
                raise MCPError(code=-32003, message="Subscription not found or access denied")
            channel = self.subscriptions[subscription_id]["channel"]
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(f"channel:{channel}")
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"].decode("utf-8"))
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Listen failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to listen: {str(e)}")

    async def close(self):
        await self.redis_client.aclose()

@router.post("/pubsub/subscribe")
async def subscribe_endpoint(user_id: str, channel: str, event_types: List[str]):
    manager = PubSubManager()
    try:
        result = await manager.subscribe(user_id, channel, event_types)
        return {"subscription_id": result}
    finally:
        await manager.close()

@router.post("/pubsub/publish")
async def publish_endpoint(channel: str, event_type: str, data: Dict[str, Any]):
    manager = PubSubManager()
    try:
        await manager.publish(channel, event_type, data)
        return {"status": "success"}
    finally:
        await manager.close()
