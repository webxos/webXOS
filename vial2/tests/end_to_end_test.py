import pytest
from fastapi.testclient import TestClient
from ..main import app
from ..mcp.database.neon_connection import neon_db
import asyncio
import logging

logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.mark.asyncio
async def test_end_to_end_flow():
    try:
        # Simulate vial command
        response = client.post("/mcp/api/vial/console/execute", json={"command": "/vial status", "vial_id": "vial1"})
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "success"

        # Verify NeonDB log
        await neon_db.connect()
        query = "SELECT event_data FROM vial_logs WHERE vial_id = $1"
        result = await neon_db.execute(query, "vial1")
        assert result is not None
        await neon_db.disconnect()
        logger.info("End-to-end test passed")
    except Exception as e:
        logger.error(f"End-to-end test failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #tests #mcp #end_to_end #neon #neon_mcp
