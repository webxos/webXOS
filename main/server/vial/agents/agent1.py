import logging,sqlite3,json
from datetime import datetime
from .auth_manager import AuthManager

logger=logging.getLogger(__name__)

class NomicAgent:
    """NomicAgent handles authentication-related tasks with wallet verification."""
    def __init__(self):
        """Initialize NomicAgent with AuthManager."""
        self.auth_manager=AuthManager()
        logger.info("NomicAgent initialized")

    async def process_authentication(self,api_key:str,wallet_id:str)->dict:
        """Process authentication for a wallet.

        Args:
            api_key (str): API key for authentication.
            wallet_id (str): Wallet ID for verification.

        Returns:
            dict: Authentication result with token.

        Raises:
            Exception: If authentication fails.
        """
        try:
            if not self.auth_manager.verify_api_key(api_key):
                logger.warning(f"Invalid API key for wallet {wallet_id}")
                raise Exception("Invalid API key")
            token=self.auth_manager.generate_token(api_key)
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO auth_events (api_key,wallet_id,event_type,timestamp) VALUES (?,?,?,?)",
                              (api_key,wallet_id,"login",datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Authentication successful for wallet {wallet_id}")
            return {"status":"success","token":token,"wallet_id":wallet_id}
        except Exception as e:
            logger.error(f"Authentication failed for wallet {wallet_id}: {str(e)}")
            raise Exception(f"Authentication failed: {str(e)}")
