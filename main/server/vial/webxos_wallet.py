import logging,sqlite3
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
from .mcp_auth_server import MCPAuthServer

logger=logging.getLogger(__name__)

class WalletRequest(BaseModel):
    wallet_id:str
    user_id:str
    api_key:str

class MCPWalletManager:
    """Manages wallet creation and verification with SQLite storage."""
    def __init__(self):
        """Initialize MCPWalletManager with OAuth server."""
        self.auth_server=MCPAuthServer()
        self.db_path="/app/vial_mcp.db"
        self.init_wallet_db()
        logger.info("MCPWalletManager initialized")

    def init_wallet_db(self):
        """Initialize SQLite table for wallet storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("""CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    timestamp TEXT NOT NULL)""")
                conn.commit()
                logger.info("Wallet SQLite table initialized")
        except Exception as e:
            logger.error(f"Wallet database initialization failed: {str(e)}")
            raise Exception(f"Wallet database initialization failed: {str(e)}")

    async def create_wallet(self,request:WalletRequest,access_token:str=Depends(lambda x: x)) -> dict:
        """Create a new wallet and associate it with a user and API key.

        Args:
            request (WalletRequest): Wallet data with wallet_id, user_id, and api_key.
            access_token (str): OAuth access token for verification.

        Returns:
            dict: Success message with wallet ID.

        Raises:
            HTTPException: If wallet creation or token verification fails.
        """
        try:
            if not await self.auth_server.verify_oauth_token(access_token,request.wallet_id):
                logger.warning(f"Invalid token for wallet {request.wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid access token")
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO wallets (wallet_id,user_id,api_key,timestamp) VALUES (?,?,?,?)",
                              (request.wallet_id,request.user_id,request.api_key,datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Created wallet {request.wallet_id} for user {request.user_id}")
            return {"status":"success","wallet_id":request.wallet_id}
        except Exception as e:
            logger.error(f"Wallet creation failed for wallet {request.wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Wallet creation failed: {str(e)}")

    async def verify_wallet(self,wallet_id:str,access_token:str=Depends(lambda x: x)) -> bool:
        """Verify if a wallet exists and is valid.

        Args:
            wallet_id (str): Wallet ID to verify.
            access_token (str): OAuth access token for verification.

        Returns:
            bool: True if wallet is valid, False otherwise.

        Raises:
            Exception: If wallet verification fails.
        """
        try:
            if not await self.auth_server.verify_oauth_token(access_token,wallet_id):
                logger.warning(f"Invalid token for wallet {wallet_id}")
                return False
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id FROM wallets WHERE wallet_id=?",(wallet_id,))
                if not cursor.fetchone():
                    logger.warning(f"Wallet {wallet_id} not found")
                    return False
            logger.info(f"Verified wallet {wallet_id}")
            return True
        except Exception as e:
            logger.error(f"Wallet verification failed for wallet {wallet_id}: {str(e)}")
            return False
