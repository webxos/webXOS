import logging,sqlite3
from datetime import datetime

logger=logging.getLogger(__name__)

class JinaAIAgent:
    """JinaAIAgent handles resource retrieval with wallet verification."""
    def __init__(self):
        """Initialize JinaAIAgent."""
        logger.info("JinaAIAgent initialized")

    async def get_latest_resources(self,wallet_id:str)->dict:
        """Retrieve the latest 10 notes as resources for a wallet.

        Args:
            wallet_id (str): Wallet ID for verification.

        Returns:
            dict: List of latest notes.

        Raises:
            Exception: If resource retrieval fails.
        """
        try:
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE wallet_id=? ORDER BY timestamp DESC LIMIT 10",(wallet_id,))
                notes=cursor.fetchall()
                resources=[{"id":n[0],"content":n[1],"resource_id":n[2],"timestamp":n[3],"wallet_id":wallet_id} for n in notes]
                logger.info(f"Retrieved {len(resources)} resources for wallet {wallet_id}")
                return {"status":"success","resources":resources}
        except Exception as e:
            logger.error(f"Resource retrieval failed for wallet {wallet_id}: {str(e)}")
            raise Exception(f"Resource retrieval failed: {str(e)}")