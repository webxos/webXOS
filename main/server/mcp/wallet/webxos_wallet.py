import logging
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
from ..db.db_manager import DatabaseManager
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class WalletRequest(BaseModel):
    user_id: str
    wallet_id: str

class WalletUpdateRequest(BaseModel):
    wallet_id: str
    settings: dict

class MCPWalletManager:
    """Manages wallet operations for Vial MCP."""
    def __init__(self, db_manager: DatabaseManager = None, security_manager: SecurityManager = None, error_handler: ErrorHandler = None):
        """Initialize MCPWalletManager with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            security_manager (SecurityManager): Security manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.security_manager = security_manager or SecurityManager()
        self.error_handler = error_handler or ErrorHandler()
        logger.info("MCPWalletManager initialized")

    async def create_wallet(self, request: WalletRequest, access_token: str) -> dict:
        """Create a new wallet for a user.

        Args:
            request (WalletRequest): Wallet creation request.
            access_token (str): JWT access token.

        Returns:
            dict: Created wallet details.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Unauthorized wallet access"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/wallet/create", request.wallet_id, Exception(error_msg))
            wallet_id = str(uuid.uuid4())
            wallet = await self.db_manager.create_wallet(request.user_id, wallet_id)
            logger.info(f"Created wallet {wallet_id} for user {request.user_id}")
            return wallet
        except Exception as e:
            self.error_handler.handle_exception("/api/wallet/create", request.wallet_id, e)

    async def update_wallet(self, request: WalletUpdateRequest, access_token: str) -> dict:
        """Update wallet settings.

        Args:
            request (WalletUpdateRequest): Wallet update request.
            access_token (str): JWT access token.

        Returns:
            dict: Updated wallet details.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Unauthorized wallet access"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/wallet/update", request.wallet_id, Exception(error_msg))
            wallet = await self.db_manager.update_wallet(request.wallet_id, request.settings)
            logger.info(f"Updated wallet {request.wallet_id}")
            return wallet
        except Exception as e:
            self.error_handler.handle_exception("/api/wallet/update", request.wallet_id, e)
