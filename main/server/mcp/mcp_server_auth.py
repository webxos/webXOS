import logging,sqlite3
from fastapi import HTTPException,Depends
from pydantic import BaseModel
from .mcp_auth_server import MCPAuthServer
from datetime import datetime

logger=logging.getLogger(__name__)

class AuthRequest(BaseModel):
    api_key:str
    wallet_id:str

class RefreshRequest(BaseModel):
    refresh_token:str
    wallet_id:str

class MCPAuthHandler:
    """Handles authentication tasks with embedded OAuth."""
    def __init__(self):
        """Initialize MCPAuthHandler with MCPAuthServer."""
        self.auth_server=MCPAuthServer()
        logger.info("MCPAuthHandler initialized")

    async def authenticate(self,request:AuthRequest)->dict:
        """Authenticate a user with API key and wallet ID.

        Args:
            request (AuthRequest): Authentication request with api_key and wallet_id.

        Returns:
            dict: OAuth tokens and expiry.

        Raises:
            HTTPException: If authentication fails.
        """
        try:
            result=await self.auth_server.generate_oauth_token(request.api_key,request.wallet_id)
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO auth_events (api_key,wallet_id,event_type,timestamp) VALUES (?,?,?,?)",
                              (request.api_key,request.wallet_id,"login",datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Authenticated wallet {request.wallet_id}")
            return result
        except Exception as e:
            logger.error(f"Authentication failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Authentication failed: {str(e)}")

    async def refresh_token(self,request:RefreshRequest)->dict:
        """Refresh OAuth token for a wallet.

        Args:
            request (RefreshRequest): Refresh request with refresh_token and wallet_id.

        Returns:
            dict: New access token and expiry.

        Raises:
            HTTPException: If token refresh fails.
        """
        try:
            result=await self.auth_server.refresh_oauth_token(request.refresh_token,request.wallet_id)
            logger.info(f"Token refreshed for wallet {request.wallet_id}")
            return result
        except Exception as e:
            logger.error(f"Token refresh failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Token refresh failed: {str(e)}")
