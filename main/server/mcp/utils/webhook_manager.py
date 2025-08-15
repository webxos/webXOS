# main/server/mcp/utils/webhook_manager.py
from typing import Dict, Any, List
from fastapi import APIRouter, Request, HTTPException
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.cache_manager import CacheManager
import aiohttp
import logging
import hmac
import hashlib
import os
import json

router = APIRouter()
logger = logging.getLogger("mcp")

class WebhookManager:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.cache = CacheManager()
        self.webhook_secret = os.getenv("WEBHOOK_SECRET", "")
        self.webhooks = {}  # In-memory for simplicity; persist in production

    @self.metrics.track_request("register_webhook")
    async def register_webhook(self, user_id: str, url: str, events: List[str]) -> str:
        try:
            if not user_id or not url or not events:
                raise MCPError(code=-32602, message="User ID, URL, and events are required")
            webhook_id = secrets.token_hex(16)
            self.webhooks[webhook_id] = {"user_id": user_id, "url": url, "events": events}
            await self.cache.set_cache(f"webhook:{webhook_id}", self.webhooks[webhook_id])
            logger.info(f"Registered webhook {webhook_id} for user {user_id}")
            return webhook_id
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to register webhook: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to register webhook: {str(e)}")

    @self.metrics.track_request("trigger_webhook")
    async def trigger_webhook(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            if not event or not payload:
                raise MCPError(code=-32602, message="Event and payload are required")
            async with aiohttp.ClientSession() as session:
                for webhook_id, webhook in self.webhooks.items():
                    if event in webhook["events"]:
                        signature = self._generate_signature(payload)
                        headers = {"X-Webhook-Signature": signature, "Content-Type": "application/json"}
                        async with session.post(webhook["url"], headers=headers, json=payload) as response:
                            if response.status >= 400:
                                logger.warning(f"Webhook {webhook_id} failed: {response.status}")
                            else:
                                logger.info(f"Triggered webhook {webhook_id} for event {event}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to trigger webhook: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to trigger webhook: {str(e)}")

    def _generate_signature(self, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True)
        return hmac.new(
            self.webhook_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

    @self.metrics.track_request("verify_webhook")
    async def verify_webhook(self, request: Request, payload: Dict[str, Any]) -> bool:
        try:
            signature = request.headers.get("X-Webhook-Signature")
            if not signature:
                raise MCPError(code=-32003, message="Missing webhook signature")
            expected_signature = self._generate_signature(payload)
            if not hmac.compare_digest(signature, expected_signature):
                raise MCPError(code=-32003, message="Invalid webhook signature")
            logger.info("Webhook signature verified")
            return True
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Webhook verification failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Webhook verification failed: {str(e)}")

    @self.metrics.track_request("delete_webhook")
    async def delete_webhook(self, webhook_id: str, user_id: str) -> None:
        try:
            if webhook_id not in self.webhooks or self.webhooks[webhook_id]["user_id"] != user_id:
                raise MCPError(code=-32003, message="Webhook not found or access denied")
            del self.webhooks[webhook_id]
            await self.cache.delete_cache(f"webhook:{webhook_id}")
            logger.info(f"Deleted webhook {webhook_id} for user {user_id}")
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to delete webhook: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to delete webhook: {str(e)}")

    async def close(self):
        await self.cache.close()

@router.post("/webhook/register")
async def register_webhook_endpoint(user_id: str, url: str, events: List[str]):
    manager = WebhookManager()
    try:
        result = await manager.register_webhook(user_id, url, events)
        return {"webhook_id": result}
    finally:
        await manager.close()

@router.post("/webhook/trigger")
async def trigger_webhook_endpoint(event: str, payload: Dict[str, Any]):
    manager = WebhookManager()
    try:
        await manager.trigger_webhook(event, payload)
        return {"status": "success"}
    finally:
        await manager.close()
