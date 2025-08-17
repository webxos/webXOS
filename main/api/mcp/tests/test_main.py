import pytest
from fastapi.testclient import TestClient
from main import app
from config.config import DatabaseConfig
from neondatabase import AsyncClient
import asyncio
import uuid

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
async def test_neon_connectivity(db_client):
    result = await db_client.query("SELECT 1 AS test")
    assert result.rows == [(1,)]
    assert len(result.rows) == 1

@pytest.mark.asyncio
async def test_wallet_endpoint(client, db_client):
    user_id = str(uuid.uuid4())
    wallet_address = str(uuid.uuid4())
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address) VALUES ($1, $2, $3)",
        [user_id, 10.0, wallet_address]
    )
    await db_client.query(
        "INSERT INTO vials (vial_id, user_id, code, wallet_address, webxos_hash) VALUES ($1, $2, $3, $4, $5)",
        ['vial1', user_id, 'print("test")', wallet_address, 'test_hash']
    )
    
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "wallet.getVialBalance",
            "params": {"user_id": user_id, "vial_id": "vial1"},
            "id": 1
        }
    )
    assert response.status_code == 200
    assert response.json()['result']['balance'] == 2.5  # 10.0 / 4 vials

@pytest.mark.asyncio
async def test_oauth_token_exchange(client, db_client, mocker):
    user_id = str(uuid.uuid4())
    access_token = str(uuid.uuid4())
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
            json=lambda: {"id": user_id},
            raise_for_status=lambda: None
        )
    )
    
    response = client.post(
        "/mcp/execute",
        json={
            "jsonrpc": "2.0",
            "method": "auth.exchangeToken",
            "params": {"code": "test_code", "redirect_uri": "https://example.com/auth/callback"},
            "id": 1
        }
    )
    assert response.status_code == 200
    assert response.json()['result']['access_token'] == access_token
    assert response.json()['result']['user_id'] == user_id
    
    user = await db_client.query("SELECT user_id, access_token FROM users WHERE user_id = $1", [user_id])
    assert user.rows[0]['access_token'] == access_token
