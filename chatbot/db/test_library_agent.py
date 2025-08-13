import pytest
import aiohttp
from fastapi.testclient import TestClient
from db.library_agent import app, LibraryRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_nomic_vial():
    request = LibraryRequest(query="test query", vial_id="1", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/library/1", json=request.dict())
    assert response.status_code == 200
    assert "response" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "nomic_query"

@pytest.mark.asyncio
async def test_cognitallmware_vial():
    request = LibraryRequest(query="test query", vial_id="2", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/library/2", json=request.dict())
    assert response.status_code == 200
    assert "response" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "cognitallmware_query"

@pytest.mark.asyncio
async def test_invalid_vial():
    request = LibraryRequest(query="test query", vial_id="5", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/library/5", json=request.dict())
    assert response.status_code == 400
    assert "Invalid vial ID" in response.json()["detail"]
