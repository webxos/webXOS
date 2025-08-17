import pytest
from fastapi.testclient import TestClient
from main import app, sanitize_input
from config.config import DatabaseConfig
from neondatabase import AsyncClient
import uuid
import hashlib
from datetime import datetime, timedelta
import secrets

@pytest.fixture
async def db_client():
    client = AsyncClient(DatabaseConfig().database_url)
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_input_sanitization(client):
    malicious_input = {
        "jsonrpc": "2.0",
        "method": "wallet.getVialBalance",
        "params": {
            "user_id": "<script>alert('xss')</script>",
            "vial_id": "vial1; DROP TABLE users;"
        },
        "id": 1
    }
    
    response = client.post("/mcp/execute", json=malicious_input)
    assert response.status_code == 401  # Unauthorized due to missing token
    
    sanitized = sanitize_input(malicious_input["params"])
    assert sanitized["user_id"] == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    assert sanitized["vial_id"] == "vial1 DROP TABLE users"

@pytest.mark.asyncio
async def test_monitoring_kpis(db_client, client, mocker):
    user_id = str(uuid.uuid4())
    session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
    access_token = str(uuid.uuid4())
    
    # Setup user and session
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address, access_token) VALUES ($1, $2, $3, $4)",
        [user_id, 0.0, str(uuid.uuid4()), access_token]
    )
    await db_client.query(
        "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
        [session_id, user_id, datetime.utcnow() + timedelta(minutes=15)]
    )
    
    # Mock security event logging
    await db_client.query(
        "INSERT INTO security_events (event_type, user_id, created_at) VALUES ($1, $2, $3)",
        ["auth_success", user_id, datetime.utcnow()]
    )
    await db_client.query(
        "INSERT INTO security_events (event_type, user_id, created_at) VALUES ($1, $2, $3)",
        ["auth_error", user_id, datetime.utcnow()]
    )
    
    response = client.get("/monitoring/kpis?time_window_hours=24", headers={
        "Authorization": f"Bearer {access_token}",
        "X-Session-ID": session_id
    })
    assert response.status_code == 200
    kpis = response.json()
    assert "auth_success_rate" in kpis
    assert "auth_failure_count" in kpis
    assert "token_validations" in kpis
    assert "active_sessions" in kpis
    assert "anomalies_detected" in kpis
    assert kpis["auth_failure_count"] == 1
    assert kpis["active_sessions"] >= 1
