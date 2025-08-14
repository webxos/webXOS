import logging,sqlite3
from datetime import datetime

logger=logging.getLogger(__name__)

class ResourceTool:
    """Manages resource retrieval with wallet-based access control."""
    def __init__(self):
        """Initialize ResourceTool."""
        logger.info("ResourceTool initialized")

    async def get_latest_resources(self,wallet_id:str,limit:int=10)->dict:
        """Retrieve the latest resources (notes) for a wallet.

        Args:
            wallet_id (str): Wallet ID for verification.
            limit (int): Maximum number of resources to retrieve (default: 10).

        Returns:
            dict: List of latest notes as resources.

        Raises:
            Exception: If resource retrieval fails.
        """
        try:
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE wallet_id=? ORDER BY timestamp DESC LIMIT ?",(wallet_id,limit))
                notes=cursor.fetchall()
                resources=[{"id":n[0],"content":n[1],"resource_id":n[2],"timestamp":n[3],"wallet_id":wallet_id} for n in notes]
                logger.info(f"Retrieved {len(resources)} resources for wallet {wallet_id}")
                return {"status":"success","resources":resources}
        except Exception as e:
            logger.error(f"Resource retrieval failed for wallet {wallet_id}: {str(e)}")
            raise Exception(f"Resource retrieval failed: {str(e)}")
