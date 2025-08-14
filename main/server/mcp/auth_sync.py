import logging
import redis
import os
import json
from datetime import datetime
from fastapi import HTTPException
from .auth_manager import AuthManager

logger = logging.getLogger(__name__)

class AuthSync:
    """Synchronizes authentication state across Vial MCP servers."""
    def __init__(self):
        """Initialize AuthSync with Redis connection."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.auth_manager = AuthManager()
        logger.info("AuthSync initialized")

    def sync_token(self, wallet_id: str, token: str, ttl: int = 3600) -> None:
        """Synchronize a JWT token across servers.

        Args:
            wallet_id (str): Wallet ID associated with the token.
            token (str): JWT token to synchronize.
            ttl (int): Time-to-live in seconds (default: 3600).

        Raises:
            HTTPException: If synchronization fails.
        """
        try:
            key = f"auth:{wallet_id}"
            self.redis_client.setex(key, ttl, token)
            logger.info(f"Synchronized token for wallet {wallet_id}")
        except Exception as e:
            logger.error(f"Token sync failed for wallet {wallet_id}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuthSync] Token sync failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Token sync failed: {str(e)}")

    def get_synced_token(self, wallet_id: str) -> str | None:
        """Retrieve a synchronized JWT token.

        Args:
            wallet_id (str): Wallet ID to retrieve token for.

        Returns:
            str | None: Synchronized token or None if not found.

        Raises:
            HTTPException: If retrieval fails.
        """
        try:
            key = f"auth:{wallet_id}"
            token = self.redis_client.get(key)
            if token:
                logger.info(f"Retrieved synced token for wallet {wallet_id}")
                return token
            logger.info(f"No synced token found for wallet {wallet_id}")
            return None
        except Exception as e:
            logger.error(f"Token retrieval failed for wallet {wallet_id}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuthSync] Token retrieval failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Token retrieval failed: {str(e)}")

    def verify_synced_token(self, wallet_id: str, token: str) -> bool:
        """Verify a token against the synchronized state.

        Args:
            wallet_id (str): Wallet ID to verify token for.
            token (str): JWT token to verify.

        Returns:
            bool: True if token matches the synchronized state, False otherwise.

        Raises:
            HTTPException: If verification fails.
        """
        try:
            synced_token = self.get_synced_token(wallet_id)
            if synced_token and synced_token == token:
                self.auth_manager.verify_token(token)  # Additional JWT validation
                logger.info(f"Verified synced token for wallet {wallet_id}")
                return True
            logger.warning(f"Invalid or missing synced token for wallet {wallet_id}")
            return False
        except Exception as e:
            logger.error(f"Token verification failed for wallet {wallet_id}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuthSync] Token verification failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Token verification failed: {str(e)}")
