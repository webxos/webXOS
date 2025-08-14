import logging,sqlite3
from fastapi import HTTPException,Depends
from pydantic import BaseModel
from .mcp_auth_server import MCPAuthServer
from datetime import datetime

logger=logging.getLogger(__name__)

class ResourceRequest(BaseModel):
    wallet_id:str
    limit:int=10

class MCPResourcesHandler:
    """Manages resource retrieval with wallet-based access control."""
    def __init__(self):
        """Initialize MCPResourcesHandler with OAuth server."""
        self.auth_server=MCPAuthServer()
        logger.info("MCPResourcesHandler initialized")

    async def get_latest_resources(self,request:ResourceRequest,access_token:str=Depends(lambda x: x)) -> dict:
        """Retrieve the latest resources (notes) for a wallet.

        Args:
            request (ResourceRequest): Request with wallet_id and optional limit.
            access_token (str): OAuth access token for verification.

        Returns:
            dict: List of latest notes as resources.

        Raises:
            HTTPException: If resource retrieval or token verification fails.
        """
        try:
            if not await self.auth_server.verify_oauth_token(access_token,request.wallet_id):
                logger.warning(f"Invalid token for wallet {request.wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid access token")
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE wallet_id=? ORDER BY timestamp DESC LIMIT ?",
                              (request.wallet_id,request.limit))
                notes=cursor.fetchall()
                resources=[{"id":n[0],"content":n[1],"resource_id":n[2],"timestamp":n[3],"wallet_id":request.wallet_id} for n in notes]
                logger.info(f"Retrieved {len(resources)} resources for wallet {request.wallet_id}")
                return {"status":"success","resources":resources}
        except Exception as e:
            logger.error(f"Resource retrieval failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Resource retrieval failed: {str(e)}")
