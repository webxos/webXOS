import pytest
from fastapi.testclient import TestClient
from db.translator_agent import app, TranslatorRequest

client = TestClient(app)

@pytest.mark.asyncio
async def test_translation():
    request = TranslatorRequest(text="Hello world", target_language="fr", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/translate", json=request.dict())
    assert response.status_code == 200
    assert "translated_text" in response.json()
    assert "wallet" in response.json()
    assert len(response.json()["wallet"]["transactions"]) == 1
    assert response.json()["wallet"]["transactions"][0]["type"] == "translation"

@pytest.mark.asyncio
async def test_invalid_language():
    request = TranslatorRequest(text="Hello world", target_language="xx", wallet={"webxos": 0.0, "transactions": []})
    response = client.post("/api/translate", json=request.dict())
    assert response.status_code == 500
    assert "Translation error" in response.json()["detail"]
