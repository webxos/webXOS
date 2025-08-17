import pytest
from fastapi.testclient import TestClient
from main import app
from config.config import DatabaseConfig
from neondatabase import AsyncClient
import uuid
import secrets
from datetime import datetime, timedelta

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
async def test_data_erasure(db_client, client):
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
    await db_client.query(
        "INSERT INTO vials (vial_id, user_id, code, wallet_address, webxos_hash) VALUES ($1, $2, $3, $4, $5)",
        ["vial1", user_id, "test_code", str(uuid.uuid4()), str(uuid.uuid4())]
    )
    await db_client.query(
        "INSERT INTO transactions (transaction_id, user_id, amount, destination_address, timestamp) VALUES ($1, $2, $3, $4, $5)",
        [str(uuid.uuid4()), user_id, 50.0, str(uuid.uuid4()), datetime.utcnow()]
    )
    
    response = client.post(
        "/privacy/erase",
        json={"user_id": user_id},
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Session-ID": session_id
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert set(result["deleted_tables"]) == {"users", "sessions", "vials", "transactions", "security_events", "quantum_links"}
    
    # Verify data deletion
    user = await db_client.query("SELECT * FROM users WHERE user_id = $1", [user_id])
    assert not user.rows
    session = await db_client.query("SELECT * FROM sessions WHERE user_id = $1", [user_id])
    assert not session.rows
    vial = await db_client.query("SELECT * FROM vials WHERE user_id = $1", [user_id])
    assert not vial.rows
    transaction = await db_client.query("SELECT * FROM transactions WHERE user_id = $1", [user_id])
    assert not transaction.rows

@pytest.mark.asyncio
async def test_data_erasure_invalid_user(db_client, client):
    user_id = str(uuid.uuid4())
    response = client.post(
        "/privacy/erase",
        json={"user_id": user_id},
        headers={
            "Authorization": f"Bearer {str(uuid.uuid4())}",
            "X-Session-ID": f"{user_id}:{secrets.token_urlsafe(32)}"
        }
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"
