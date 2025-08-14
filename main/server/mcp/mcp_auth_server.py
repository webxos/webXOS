import logging,sqlite3,os,jwt
from fastapi import HTTPException
from datetime import datetime,timedelta
from .auth_manager import AuthManager

logger=logging.getLogger(__name__)

class MCPAuthServer:
    """Handles embedded OAuth authentication with wallet verification."""
    def __init__(self):
        """Initialize MCPAuthServer with AuthManager and SQLite."""
        self.auth_manager=AuthManager()
        self.db_path="/app/vial_mcp.db"
        self.init_oauth_db()
        logger.info("MCPAuthServer initialized")

    def init_oauth_db(self):
        """Initialize SQLite tables for OAuth tokens and sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("""CREATE TABLE IF NOT EXISTS oauth_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    timestamp TEXT NOT NULL)""")
                cursor.execute("""CREATE TABLE IF NOT EXISTS oauth_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id TEXT NOT NULL,
                    session_token TEXT NOT NULL,
                    timestamp TEXT NOT NULL)""")
                conn.commit()
                logger.info("OAuth SQLite tables initialized")
        except Exception as e:
            logger.error(f"OAuth database initialization failed: {str(e)}")
            raise Exception(f"OAuth database initialization failed: {str(e)}")

    async def generate_oauth_token(self,api_key:str,wallet_id:str)->dict:
        """Generate OAuth access and refresh tokens for a wallet.

        Args:
            api_key (str): API key for verification.
            wallet_id (str): Wallet ID for access control.

        Returns:
            dict: OAuth tokens and expiry.

        Raises:
            HTTPException: If token generation fails.
        """
        try:
            if not self.auth_manager.verify_api_key(api_key):
                logger.warning(f"Invalid API key for wallet {wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid API key")
            access_token=self.auth_manager.generate_token(api_key)
            refresh_token=jwt.encode({"wallet_id":wallet_id,"exp":datetime.utcnow()+timedelta(days=7)},
                                    self.auth_manager.secret_key,algorithm="HS256")
            expires_at=(datetime.utcnow()+timedelta(hours=1)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT INTO oauth_tokens (wallet_id,access_token,refresh_token,expires_at,timestamp) VALUES (?,?,?,?,?)",
                              (wallet_id,access_token,refresh_token,expires_at,datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Generated OAuth tokens for wallet {wallet_id}")
            return {"access_token":access_token,"refresh_token":refresh_token,"expires_at":expires_at}
        except Exception as e:
            logger.error(f"OAuth token generation failed for wallet {wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"OAuth token generation failed: {str(e)}")

    async def refresh_oauth_token(self,refresh_token:str,wallet_id:str)->dict:
        """Refresh OAuth access token using refresh token.

        Args:
            refresh_token (str): Refresh token to validate.
            wallet_id (str): Wallet ID for verification.

        Returns:
            dict: New access token and expiry.

        Raises:
            HTTPException: If token refresh fails.
        """
        try:
            payload=jwt.decode(refresh_token,self.auth_manager.secret_key,algorithms=["HS256"])
            if payload["wallet_id"]!=wallet_id:
                logger.warning(f"Invalid refresh token for wallet {wallet_id}")
                raise HTTPException(status_code=401,detail="Invalid refresh token")
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id FROM oauth_tokens WHERE refresh_token=? AND wallet_id=?",(refresh_token,wallet_id))
                if not cursor.fetchone():
                    logger.warning(f"Refresh token not found for wallet {wallet_id}")
                    raise HTTPException(status_code=401,detail="Refresh token not found")
            api_key=next(k for k,v in self.auth_manager.api_keys.items() if v["wallet_id"]==wallet_id)
            access_token=self.auth_manager.generate_token(api_key)
            expires_at=(datetime.utcnow()+timedelta(hours=1)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("UPDATE oauth_tokens SET access_token=?,expires_at=? WHERE refresh_token=?",
                              (access_token,expires_at,refresh_token))
                conn.commit()
            logger.info(f"Refreshed OAuth token for wallet {wallet_id}")
            return {"access_token":access_token,"expires_at":expires_at}
        except jwt.ExpiredSignatureError:
            logger.warning(f"Refresh token expired for wallet {wallet_id}")
            raise HTTPException(status_code=401,detail="Refresh token expired")
        except Exception as e:
            logger.error(f"OAuth token refresh failed for wallet {wallet_id}: {str(e)}")
            raise HTTPException(status_code=500,detail=f"OAuth token refresh failed: {str(e)}")

    async def verify_oauth_token(self,access_token:str,wallet_id:str)->bool:
        """Verify OAuth access token for a wallet.

        Args:
            access_token (str): Access token to verify.
            wallet_id (str): Wallet ID for verification.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            Exception: If token verification fails.
        """
        try:
            payload=self.auth_manager.verify_token(access_token)
            if payload["wallet_id"]!=wallet_id:
                logger.warning(f"Token wallet mismatch for {wallet_id}")
                return False
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("SELECT id FROM oauth_tokens WHERE access_token=? AND wallet_id=?",(access_token,wallet_id))
                if not cursor.fetchone():
                    logger.warning(f"Access token not found for wallet {wallet_id}")
                    return False
            logger.info(f"Verified OAuth token for wallet {wallet_id}")
            return True
        except Exception as e:
            logger.error(f"OAuth token verification failed for wallet {wallet_id}: {str(e)}")
            return False
