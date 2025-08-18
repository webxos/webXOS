import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..error_logging.error_log import error_logger
import logging
import sqlite3

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_offline_cache():
    try:
        response = client.post("/mcp/api/vial/offline", json={"vial_id": "vial1", "action": "cache", "data": {"task": "test_task"}})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["action"] == "cache"
        with sqlite3.connect("error_log.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM offline_queue WHERE vial_id = 'vial1'").fetchone()[0]
            assert count == 1
    except Exception as e:
        error_logger.log_error("test_offline_cache", str(e), str(e.__traceback__), sql_statement="INSERT INTO offline_queue", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Offline cache test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_offline_sync():
    try:
        # Pre-populate queue
        with sqlite3.connect("error_log.db") as conn:
            conn.execute("INSERT INTO offline_queue (vial_id, action, data) VALUES (?, ?, ?)",
                        ("vial1", "cache", '{"task": "test_task"}'))
        response = client.post("/mcp/api/vial/offline", json={"vial_id": "vial1", "action": "sync"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"
        assert response.json()["result"]["action"] == "sync"
        with sqlite3.connect("error_log.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM offline_queue WHERE vial_id = 'vial1'").fetchone()[0]
            assert count == 0
    except Exception as e:
        error_logger.log_error("test_offline_sync", str(e), str(e.__traceback__), sql_statement="SELECT FROM offline_queue", sql_error_code=None, params={"vial_id": "vial1"})
        logger.error(f"Offline sync test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #offline #sqlite #neon_mcp
