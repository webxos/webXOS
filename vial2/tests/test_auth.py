import pytest
import asyncpg
from ..auth import handle_auth, generate_api_key
from ..config import config
from ..error_logging.error_log import error_logger

@pytest.mark.asyncio
async def test_handle_auth():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        request = {
            "method": "authenticate",
            "code": "test_code",
            "redirect_uri": "https://webxos.netlify.app/callback"
        }
        with pytest.raises(Exception):  # Mocking external auth failure
            await handle_auth("authenticate", request)
    except Exception as e:
        error_logger.log_error("test_auth", f"Test handle_auth failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_generate_api_key():
    try:
        db = await asyncpg.connect(config.DATABASE_URL)
        user_id = "0x1234567890abcdef1234567890abcdef12345678"
        await db.execute(
            "INSERT INTO users (wallet_address) VALUES ($1) ON CONFLICT DO NOTHING",
            user_id
        )
        result = await generate_api_key(user_id)
        assert "api_key" in result
        assert "api_password" in result
    except Exception as e:
        error_logger.log_error("test_auth", f"Test generate_api_key failed: {str(e)}", str(e.__traceback__))
        raise
    finally:
        await db.close()

# xAI Artifact Tags: #vial2 #tests #auth #neon_mcp
