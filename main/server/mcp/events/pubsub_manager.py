import logging
import redis.asyncio as redis
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class PubSubRequest(BaseModel):
    channel: str
    message: str

class PubSubManager:
    """Manages Redis-based pub/sub for event-driven communication."""
    def __init__(self, error_handler: ErrorHandler = None):
        """Initialize PubSubManager with Redis connection.

        Args:
            error_handler (ErrorHandler): Error handler instance.
        """
        self.redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)
        self.error_handler = error_handler or ErrorHandler()
        logger.info("PubSubManager initialized")

    async def publish_event(self, request: PubSubRequest) -> dict:
        """Publish an event to a Redis channel.

        Args:
            request (PubSubRequest): Publish request with channel and message.

        Returns:
            dict: Publish result.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            await self.redis_client.publish(request.channel, request.message)
            logger.info(f"Published event to channel {request.channel}")
            return {"status": "success", "channel": request.channel, "message": request.message}
        except Exception as e:
            self.error_handler.handle_exception("/api/events/publish", request.channel, e)

    async def subscribe_channel(self, channel: str) -> dict:
        """Subscribe to a Redis channel and return the latest message.

        Args:
            channel (str): Channel to subscribe to.

        Returns:
            dict: Latest message from the channel.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(channel)
            message = await pubsub.get_message(timeout=5.0)
            if message and message["type"] == "message":
                logger.info(f"Received message from channel {channel}")
                return {"channel": channel, "message": message["data"]}
            return {"channel": channel, "message": None}
        except Exception as e:
            self.error_handler.handle_exception("/api/events/subscribe", channel, e)
