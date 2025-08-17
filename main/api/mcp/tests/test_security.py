import pytest
from fastapi.testclient import TestClient
from main import app
from config.config import DatabaseConfig
from neondatabase import AsyncClient
import uuid
import secrets
from datetime import datetime, timedelta
import json

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
async def test_audit_logging(db_client, client):
    user_id = str(uuid.uuid4())
    session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
    access_token = str(uuid.uuid4())
    
    # Setup test data
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address, access_token) VALUES ($1, $2, $3, $4)",
        [user_id, 100.0, str(uuid.uuid4()), access_token]
    )
    await db_client.query(
        "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
        [session_id, user_id, datetime.utcnow() + timedelta(minutes=15)]
    )
    
    # Trigger actions to generate audit logs
    # 1. Session validation
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "wallet.getVialBalance",
            "params": {"user_id": user_id, "vial_id": "vial1"},
            "id": 1
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Session-ID": session_id
        }
    )
    assert response.status_code == 200
    
    # 2. Data erasure
    response = client.post(
        "/privacy/erase",
        json={"user_id": user_id},
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Session-ID": session_id
        }
    )
    assert response.status_code == 200
    
    # Verify audit logs
    audit_logs = await db_client.query(
        "SELECT action, details FROM audit_logs WHERE user_id = $1",
        [user_id]
    )
    assert len(audit_logs.rows) >= 2
    actions = [row["action"] for row in audit_logs.rows]
    assert "session_validate" in actions
    assert "data_erasure" in actions
    
    session_log = next(row for row in audit_logs.rows if row["action"] == "session_validate")
    assert json.loads(session_log["details"])["session_id"] == session_id
    
    erasure_log = next(row for row in audit_logs.rows if row["action"] == "data_erasure")
    assert "deleted_tables" in json.loads(erasure_log["details"])

@pytest.mark.asyncio
async def test_anomaly_logging(db_client, client):
    user_id = str(uuid.uuid4())
    session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
    access_token = str(uuid.uuid4())
    
    # Setup test data
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address, access_token) VALUES ($1, $2, $3, $4)",
        [user_id, 100.0, str(uuid.uuid4()), access_token]
    )
    await db_client.query(
        "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
        [session_id, user_id, datetime.utcnow() + timedelta(minutes=15)]
    )
    
    # Simulate multiple auth failures to trigger anomaly
    for _ in range(6):
        await db_client.query(
            "INSERT INTO security_events (event_type, user_id, created_at) VALUES ($1, $2, $3)",
            ["auth_error", user_id, datetime.utcnow()]
        )
    
    # Trigger API request to check anomaly detection
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "wallet.getVialBalance",
            "params": {"user_id": user_id, "vial_id": "vial1"},
            "id": 1
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Session-ID": session_id
        }
    )
    assert response.status_code == 200
    
    # Verify anomaly audit log
    audit_logs = await db_client.query(
        "SELECT action, details FROM audit_logs WHERE user_id = $1 AND action = $2",
        [user_id, "anomaly_detected"]
    )
    assert len(audit_logs.rows) == 1
    assert json.loads(audit_logs.rows[0]["details"])["type"] == "auth_failure_rate"
    assert json.loads(audit_logs.rows[0]["details"])["count"] > 5
