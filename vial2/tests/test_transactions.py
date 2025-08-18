import pytest
import asyncpg
from ..wallet.transactions import process_transaction
from ..config import config
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_process_transaction():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
        await db.execute(
            "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
            wallet_address
        )
        await db.execute(
            "INSERT INTO wallets (user_id, address, balance, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3, $4)",
            wallet_address, wallet_address, 100.0, "test_hash"
        )
        result = await process_transaction(wallet_address, 50.0, "credit")
        assert result["status"] == "success"
        assert result["new_balance"] == 150.0
    except Exception as e:
        error_logger.log_error("test_transactions", f"Test process_transaction failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_insufficient_balance():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        wallet_address = "0x9876543210fedcba9876543210fedcba98765432"
        await db.execute(
            "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
            wallet_address
        )
        await db.execute(
            "INSERT INTO wallets (user_id, address, balance, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3, $4)",
            wallet_address, wallet_address, 10.0, "test_hash"
        )
        with pytest.raises(HTTPException) as exc:
            await process_transaction(wallet_address, 50.0, "debit")
        assert exc.value.status_code == 400
        assert "Insufficient balance" in str(exc.value.detail)
    except Exception as e:
        error_logger.log_error("test_transactions", f"Test insufficient_balance failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #tests #transactions #neon_mcp
