# main/server/mcp/db/test_db_manager.py
import pytest
from .db_manager import DBManager

@pytest.fixture
def db_manager():
    return DBManager()

@pytest.mark.asyncio
async def test_save_user(db_manager):
    result = await db_manager.save_user("test_user", "test_pass", {"vial1": {"balance": 100}})
    assert result is True
