import pytest
from fastapi.testclient import TestClient
from db.monitor_agent import app, MonitorRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_monitor_libraries():
    request = MonitorRequest(user_id="test_user", vials=["1", "3"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/monitor", json=request.dict())
    assert response.status_code == 200
    assert "results" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "monitor"

@pytest.mark.asyncio
async def test_invalid_vial_monitor():
    request = MonitorRequest(user_id="test_user", vials=["5"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/monitor", json=request.dict())
    assert response.status_code == 200
    assert len(response.json()["results"]) == 0
    assert len(response.json()["wallet"]["transactions"]) == 1
