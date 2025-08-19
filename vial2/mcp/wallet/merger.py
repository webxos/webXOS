from mcp.wallet.serializer import wallet_serializer
from mcp.wallet.validator import wallet_validator
from mcp.database.neon_connection import neon_db
from mcp.error_logging.error_log import error_logger
import logging
import json

logger = logging.getLogger(__name__)

class WalletMerger:
    async def merge_wallets(self, wallet_id1: str, wallet_id2: str, token: str):
        try:
            if not wallet_validator.validate_wallet(wallet_id1, token) or not wallet_validator.validate_wallet(wallet_id2, token):
                raise ValueError("Wallet validation failed")
            serialized1 = wallet_serializer.serialize_wallet({"id": wallet_id1})
            serialized2 = wallet_serializer.serialize_wallet({"id": wallet_id2})
            merged = {**serialized1, **serialized2}
            query = "INSERT INTO vial_users (wallet_id, user_data) VALUES ($1, $2) ON CONFLICT (wallet_id) DO UPDATE SET user_data = $2"
            await neon_db.execute(query, f"{wallet_id1}_{wallet_id2}", json.dumps(merged))
            logger.info(f"Merged wallets {wallet_id1} and {wallet_id2}")
            return merged
        except Exception as e:
            error_logger.log_error("wallet_merge", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={wallet_id1, wallet_id2})
            logger.error(f"Wallet merge failed: {str(e)}")
            raise

wallet_merger = WalletMerger()

# xAI Artifact Tags: #vial2 #mcp #wallet #merger #security #neon_mcp
