import pytest
from fastapi.testclient import TestClient
from db.config_validator import app, ConfigValidateRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_validate_config():
    request = ConfigValidateRequest(user_id="test_user", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/validate_config", json=request.dict())
    assert response.status_code == 200
    assert "result" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "config_validation"

@pytest.mark.asyncio
async def test_invalid_config_file():
    import os
    os.rename("db/library_config.json", "db/library_config.json.bak")
    request = ConfigValidateRequest(user_id="test_user", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/validate_config", json=request.dict())
    assert response.status_code == 500
    assert "Config validation error" in response.json()["detail"]
    os.rename("db/library_config.json.bak", "db/library_config.json")
