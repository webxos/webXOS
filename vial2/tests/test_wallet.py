import pytest
import asyncpg
from ..wallet import validate_wallet
from ..wallet_export import export_wallet, import_wallet
from ..config import config
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_validate_wallet():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        wallet_data = {"address": "0x1234567890abcdef1234567890abcdef12345678"}
        result = await validate_wallet(wallet_data, db)
        assert result["status"] == "success"
        assert result["wallet_address"] == wallet_data["address"]
        assert len(result["hash"]) == 64
    except Exception as e:
        error_logger.log_error("test_wallet", f"Test validate_wallet failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_export_wallet():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        wallet_address = "0x1234567890abcdef1234567890abcdef12345678"
        await db.execute(
            "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
            wallet_address
        )
        await db.execute(
            "INSERT INTO wallets (user_id, address, hash) VALUES ((SELECT id FROM users WHERE wallet_address=$1), $2, $3)",
            wallet_address, wallet_address, "test_hash"
        )
        result = await export_wallet(wallet_address, db)
        assert result["status"] == "success"
        assert result["wallet_address"] == wallet_address
    except Exception as e:
        error_logger.log_error("test_wallet", f"Test export_wallet failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_import_wallet():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        wallet_data = {
            "address": "0x9876543210fedcba9876543210fedcba98765432",
            "balance": 0.0,
            "hash": "test_hash_import",
            "created_at": "2025-08-18T09:00:00"
        }
        with open("wallets/test_wallet.md", "w") as f:
            import json
            json.dump(wallet_data, f)
        result = await import_wallet("wallets/test_wallet.md", db)
        assert result["status"] == "success"
        assert result["wallet_address"] == wallet_data["address"]
    except Exception as e:
        error_logger.log_error("test_wallet", f"Test import_wallet failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #tests #wallet #neon_mcp
