import logging
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..db.db_manager import DatabaseManager
from ..cache_manager import CacheManager
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class NoteRequest(BaseModel):
    wallet_id: str
    content: str
    resource_id: str
    db_type: str

class NoteReadRequest(BaseModel):
    wallet_id: str
    db_type: str

class MCPNotesHandler:
    """Handles note-related operations for Vial MCP."""
    def __init__(self, db_manager: DatabaseManager = None, cache_manager: CacheManager = None, security_manager: SecurityManager = None, error_handler: ErrorHandler = None):
        """Initialize MCPNotesHandler with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            cache_manager (CacheManager): Cache manager instance.
            security_manager (SecurityManager): Security manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.cache_manager = cache_manager or CacheManager()
        self.security_manager = security_manager or SecurityManager()
        self.error_handler = error_handler or ErrorHandler()
        logger.info("MCPNotesHandler initialized")

    async def add_note(self, request: NoteRequest, access_token: str) -> dict:
        """Add a new note for a wallet.

        Args:
            request (NoteRequest): Note creation request.
            access_token (str): JWT access token.

        Returns:
            dict: Added note details.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Unauthorized wallet access"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/notes/add", request.wallet_id, Exception(error_msg))
            note = await self.db_manager.add_note(request.wallet_id, request.content, request.resource_id, request.db_type)
            cache_key = f"notes:{request.wallet_id}:{request.db_type}"
            self.cache_manager.invalidate_cache(cache_key)
            logger.info(f"Added note for wallet {request.wallet_id} in {request.db_type}")
            return note
        except Exception as e:
            self.error_handler.handle_exception("/api/notes/add", request.wallet_id, e)

    async def read_note(self, request: NoteReadRequest, access_token: str) -> dict:
        """Retrieve notes for a wallet.

        Args:
            request (NoteReadRequest): Note retrieval request.
            access_token (str): JWT access token.

        Returns:
            dict: List of notes.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Unauthorized wallet access"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/notes/read", request.wallet_id, Exception(error_msg))
            cache_key = f"notes:{request.wallet_id}:{request.db_type}"
            cached = self.cache_manager.get_cached_response(cache_key)
            if cached:
                logger.info(f"Cache hit for notes: {cache_key}")
                return {"notes": cached}
            notes = await self.db_manager.get_notes(request.wallet_id, 10, request.db_type)
            self.cache_manager.cache_response(cache_key, notes)
            logger.info(f"Retrieved {len(notes)} notes for wallet {request.wallet_id} from {request.db_type}")
            return {"notes": notes}
        except Exception as e:
            self.error_handler.handle_exception("/api/notes/read", request.wallet_id, e)
