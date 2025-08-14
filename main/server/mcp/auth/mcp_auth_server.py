import logging
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..db.db_manager import DatabaseManager
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class AuthRequest(BaseModel):
    wallet_id: str
    api_key: str

class MCPAuthServer:
    """Handles authentication operations for Vial MCP."""
    def __init__(self, db_manager: DatabaseManager = None, security_manager: SecurityManager = None, error_handler: ErrorHandler = None):
        """Initialize MCPAuthServer with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            security_manager (SecurityManager): Security manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.security_manager = security_manager or SecurityManager()
        self.error_handler = error_handler or ErrorHandler()
        logger.info("MCPAuthServer initialized")

    async def authenticate(self, request: AuthRequest) -> dict:
        """Authenticate a user and issue a JWT token.

        Args:
            request (AuthRequest): Authentication request with wallet_id and api_key.

        Returns:
            dict: Authentication response with access token.

        Raises:
            HTTPException: If authentication fails.
        """
        try:
            # Validate API key against database
            user = await self.db_manager.get_user(request.wallet_id, request.api_key)
            if not user:
                error_msg = f"Invalid credentials for wallet {request.wallet_id}"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/auth/login", request.wallet_id, Exception(error_msg))
            # Generate JWT token
            token = self.security_manager.generate_token({"wallet_id": request.wallet_id})
            logger.info(f"Authenticated wallet {request.wallet_id}")
            return {"access_token": token, "token_type": "bearer"}
        except Exception as e:
            self.error_handler.handle_exception("/api/auth/login", request.wallet_id, e)
