import logging
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..db.db_manager import DatabaseManager
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class AuthSyncRequest(BaseModel):
    wallet_id: str
    access_token: str

class AuthSyncManager:
    """Synchronizes authentication tokens across services."""
    def __init__(self, db_manager: DatabaseManager = None, security_manager: SecurityManager = None, error_handler: ErrorHandler = None):
        """Initialize AuthSyncManager with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            security_manager (SecurityManager): Security manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.security_manager = security_manager or SecurityManager()
        self.error_handler = error_handler or ErrorHandler()
        logger.info("AuthSyncManager initialized")

    async def sync_auth_token(self, request: AuthSyncRequest) -> dict:
        """Synchronize authentication token for a wallet.

        Args:
            request (AuthSyncRequest): Token synchronization request.

        Returns:
            dict: Synchronization result.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(request.access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Invalid token for wallet"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/sync/auth", request.wallet_id, Exception(error_msg))
            await self.db_manager.update_token(request.wallet_id, request.access_token)
            logger.info(f"Synchronized token for wallet {request.wallet_id}")
            return {"status": "success", "wallet_id": request.wallet_id}
        except Exception as e:
            self.error_handler.handle_exception("/api/sync/auth", request.wallet_id, e)
