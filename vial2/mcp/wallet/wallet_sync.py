from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class WalletSync:
    async def sync_wallet(self, wallet_id: str):
        try:
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, wallet_id, "wallet_sync", {"status": "synced", "balance": os.getenv("WALLET_BALANCE", "0")})
            logger.info(f"Synced wallet {wallet_id}")
        except Exception as e:
            error_logger.log_error("wallet_sync", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={wallet_id})
            logger.error(f"Wallet sync failed: {str(e)}")
            raise

wallet_sync = WalletSync()

# xAI Artifact Tags: #vial2 #mcp #wallet #sync #neon #neon_mcp
