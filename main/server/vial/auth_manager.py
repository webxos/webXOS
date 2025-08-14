import jwt,hashlib,logging
from datetime import datetime,timedelta
from fastapi import HTTPException

logger=logging.getLogger(__name__)

class AuthManager:
    """Manages authentication with API keys and JWT tokens, tied to wallet IDs."""
    def __init__(self):
        """Initialize AuthManager with secret key and API key mappings."""
        self.secret_key="vial-mcp-secret-2025"
        self.api_keys={
            "api-a24cb96b-96cd-488d-a013-91cb8edbbe68":{"user_id":"vial_user","wallet_id":"wallet_123","created_at":datetime.now()},
            "api-bd9d62ec-a074-4548-8c83-fb054715a870":{"user_id":"vial_user","wallet_id":"wallet_456","created_at":datetime.now()}
        }

    def verify_api_key(self,api_key:str)->bool:
        """Verify if the API key is valid and associated with a wallet.

        Args:
            api_key (str): API key to verify.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            if api_key not in self.api_keys:
                logger.warning(f"Invalid API key attempted: {api_key}")
                return False
            return True
        except Exception as e:
            logger.error(f"API key verification failed: {str(e)}")
            return False

    def generate_token(self,api_key:str)->str:
        """Generate a JWT token for a valid API key.

        Args:
            api_key (str): API key to generate token for.

        Returns:
            str: JWT token.

        Raises:
            HTTPException: If token generation fails.
        """
        try:
            if not self.verify_api_key(api_key):
                raise HTTPException(status_code=401,detail="Invalid API key")
            payload={"user_id":self.api_keys[api_key]["user_id"],"wallet_id":self.api_keys[api_key]["wallet_id"],"exp":datetime.utcnow()+timedelta(hours=24)}
            token=jwt.encode(payload,self.secret_key,algorithm="HS256")
            logger.info(f"Generated token for wallet {self.api_keys[api_key]['wallet_id']}")
            return token
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Token generation failed: {str(e)}")

    def verify_token(self,token:str)->dict:
        """Verify a JWT token and return its payload.

        Args:
            token (str): JWT token to verify.

        Returns:
            dict: Decoded token payload.

        Raises:
            HTTPException: If token is invalid or expired.
        """
        try:
            payload=jwt.decode(token,self.secret_key,algorithms=["HS256"])
            logger.info(f"Token verified for wallet {payload['wallet_id']}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=401,detail="Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise HTTPException(status_code=401,detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(status_code=500,detail=f"Token verification failed: {str(e)}")
