import pytest
from fastapi.testclient import TestClient
from db.library_sync import app, LibrarySyncRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_library_sync():
    request = LibrarySyncRequest(user_id="test_user", vials=["1", "2"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/sync_vials", json=request.dict())
    assert response.status_code == 200
    assert "results" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "library_sync"

@pytest.mark.asyncio
async def test_invalid_vial_sync():
    request = LibrarySyncRequest(user_id="test_user", vials=["5"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/sync_vials", json=request.dict())
    assert response.status_code == 200
    assert len(response.json()["results"]) == 0
    assert len(response.json()["wallet"]["transactions"]) == 1
