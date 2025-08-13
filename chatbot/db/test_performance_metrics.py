import pytest
from fastapi.testclient import TestClient
from db.performance_metrics import app, MetricsRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_collect_metrics():
    request = MetricsRequest(user_id="test_user", vials=["1", "2"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/collect_metrics", json=request.dict())
    assert response.status_code == 200
    assert "metrics" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "metrics_collection"

@pytest.mark.asyncio
async def test_invalid_vial_metrics():
    request = MetricsRequest(user_id="test_user", vials=["5"], wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/collect_metrics", json=request.dict())
    assert response.status_code == 200
    assert len(response.json()["metrics"]) == 0
    assert len(response.json()["wallet"]["transactions"]) == 1
