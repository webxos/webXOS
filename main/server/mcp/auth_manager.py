import logging
import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages JWT-based authentication for Vial MCP."""
    def __init__(self):
        """Initialize AuthManager with JWT secret."""
        self.secret_key = os.getenv("JWT_SECRET", "your_jwt_secret")
        self.algorithm = "HS256"
        logger.info("AuthManager initialized")

    def create_token(self, wallet_id: str) -> dict:
        """Create a JWT access token for a wallet.

        Args:
            wallet_id (str): Wallet ID to encode in the token.

        Returns:
            dict: Access token and expiry details.
        """
        try:
            payload = {
                "wallet_id": wallet_id,
                "exp": datetime.utcnow() + timedelta(hours=1),
                "iat": datetime.utcnow()
            }
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created token for wallet {wallet_id}")
            return {"access_token": token, "expires_in": 3600}
        except Exception as e:
            logger.error(f"Token creation failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuthManager] Token creation failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Token creation failed: {str(e)}")

    def verify_token(self, token: str) -> dict:
        """Verify a JWT token.

        Args:
            token (str): JWT token to verify.

        Returns:
            dict: Decoded token payload.

        Raises:
            HTTPException: If token is invalid or expired.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            logger.info(f"Verified token for wallet {payload['wallet_id']}")
            return payload
        except JWTError as e:
            logger.error(f"Token verification failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuthManager] Token verification failed: {str(e)}\n")
            raise HTTPException(status_code=401, detail="Invalid access token")
