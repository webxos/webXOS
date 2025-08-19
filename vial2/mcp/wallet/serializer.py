from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from mcp.error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class WalletSerializer:
    def __init__(self):
        self.key = os.getenv("WALLET_ENCRYPT_KEY", "").encode()

    def serialize_wallet(self, wallet_data: dict):
        try:
            cipher = AES.new(self.key, AES.MODE_CBC)
            padded_data = pad(json.dumps(wallet_data).encode(), AES.block_size)
            encrypted = cipher.encrypt(padded_data)
            iv = base64.b64encode(cipher.iv).decode('utf-8')
            encrypted_data = base64.b64encode(encrypted).decode('utf-8')
            logger.info("Wallet serialized successfully")
            return {"iv": iv, "data": encrypted_data}
        except Exception as e:
            error_logger.log_error("wallet_serialize", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Wallet serialization failed: {str(e)}")
            raise

wallet_serializer = WalletSerializer()

# xAI Artifact Tags: #vial2 #mcp #wallet #serializer #security #neon_mcp
