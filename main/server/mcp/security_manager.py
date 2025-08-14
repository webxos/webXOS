import logging
from fastapi import HTTPException
from python_jose import jwt
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages security policies and JWT token validation for Vial MCP."""
    def __init__(self):
        """Initialize SecurityManager with JWT settings."""
        self.jwt_secret = os.getenv("JWT_SECRET", "your_jwt_secret")
        self.jwt_algorithm = "HS256"
        logger.info("SecurityManager initialized")

    def create_token(self, wallet_id: str, expires_delta: timedelta = timedelta(hours=1)) -> str:
        """Create a JWT token for a wallet.

        Args:
            wallet_id (str): Wallet ID to encode in token.
            expires_delta (timedelta): Token expiration time.

        Returns:
            str: Encoded JWT token.

        Raises:
            HTTPException: If token creation fails.
        """
        try:
            to_encode = {"wallet_id": wallet_id, "exp": datetime.utcnow() + expires_delta}
            token = jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
            logger.info(f"Created JWT token for wallet {wallet_id}")
            return token
        except Exception as e:
            logger.error(f"Token creation failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [SecurityManager] Token creation failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Token creation failed: {str(e)}")

    def validate_token(self, token: str) -> dict:
        """Validate a JWT token.

        Args:
            token (str): JWT token to validate.

        Returns:
            dict: Decoded token payload.

        Raises:
            HTTPException: If token is invalid or expired.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            logger.info(f"Validated token for wallet {payload['wallet_id']}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.error("Token expired")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [SecurityManager] Token expired\n")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError as e:
            logger.error(f"Invalid token: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [SecurityManager] Invalid token: {str(e)}\n")
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
