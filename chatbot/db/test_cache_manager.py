import pytest
from fastapi.testclient import TestClient
from db.cache_manager import app, CacheRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_cache_response():
    request = CacheRequest(user_id="test_user", vial_id="1", query="test query", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/cache_response", json=request.dict())
    assert response.status_code == 200
    assert "response" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "cache_response"

@pytest.mark.asyncio
async def test_invalid_vial_cache():
    request = CacheRequest(user_id="test_user", vial_id="5", query="test query", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/cache_response", json=request.dict())
    assert response.status_code == 500
    assert "Library agent call failed" in response.json()["detail"]
