import pytest
from fastapi.testclient import TestClient
from main import app
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
async def test_generate_api_credentials(db_client, client):
    user_id = str(uuid.uuid4())
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address) VALUES ($1, $2, $3)",
        [user_id, 0.0, str(uuid.uuid4())]
    )
    
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "auth.generateAPICredentials",
            "params": {"user_id": user_id},
            "id": 1
        }
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert "api_key" in result
    assert "api_secret" in result
    
    user = await db_client.query(
        "SELECT api_key, api_secret FROM users WHERE user_id = $1",
        [user_id]
    )
    assert user.rows[0]["api_key"] == result["api_key"]
    assert user.rows[0]["api_secret"] == hashlib.sha256(result["api_secret"].encode()).hexdigest()

@pytest.mark.asyncio
async def test_exchange_token_with_pkce(db_client, client, mocker):
    user_id = str(uuid.uuid4())
    access_token = str(uuid.uuid4())
    code_verifier = secrets.token_urlsafe(32)
    
    mocker.patch(
        'httpx.AsyncClient.post',
        return_value=mocker.Mock(
            json=lambda: {"access_token": access_token},
            raise_for_status=lambda: None
        )
    )
    mocker.patch(
        'httpx.AsyncClient.get',
        return_value=mocker.Mock(
            json=lambda: {"id": user_id, "aud": "test_client_id"},
            raise_for_status=lambda: None
        )
    )
    
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "auth.exchangeToken",
            "params": {
                "code": "test_code",
                "redirect_uri": "https://webxos.netlify.app/auth/callback",
                "code_verifier": code_verifier
            },
            "id": 1
        }
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["access_token"] == access_token
    assert result["user_id"] == user_id
    assert "session_id" in result
    
    session = await db_client.query(
        "SELECT session_key, expires_at FROM sessions WHERE user_id = $1",
        [user_id]
    )
    assert session.rows[0]["session_key"] == result["session_id"]
    assert session.rows[0]["expires_at"] > datetime.utcnow()

@pytest.mark.asyncio
async def test_token_revocation(db_client, client):
    user_id = str(uuid.uuid4())
    access_token = str(uuid.uuid4())
    session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
    
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address, access_token) VALUES ($1, $2, $3, $4)",
        [user_id, 0.0, str(uuid.uuid4()), access_token]
    )
    await db_client.query(
        "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
        [session_id, user_id, datetime.utcnow() + timedelta(minutes=15)]
    )
    
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "auth.revokeToken",
            "params": {"user_id": user_id, "access_token": access_token},
            "id": 1
        },
        headers={"X-Session-ID": session_id}
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "revoked"
    
    user = await db_client.query(
        "SELECT access_token FROM users WHERE user_id = $1",
        [user_id]
    )
    assert user.rows[0]["access_token"] is None
    
    session = await db_client.query(
        "SELECT session_key FROM sessions WHERE session_key = $1",
        [session_id]
    )
    assert not session.rows
