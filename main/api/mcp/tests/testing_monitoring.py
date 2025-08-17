import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocket
from main import app
from config.config import DatabaseConfig
from neondatabase import AsyncClient
import redis.asyncio as redis
import uuid
import secrets
from datetime import datetime, timedelta
import asyncio

@pytest.fixture
async def db_client():
    client = AsyncClient(DatabaseConfig().database_url)
    await client.connect()
    yield client
    await client.disconnect()

@pytest.fixture
async def redis_client():
    client = redis.from_url("redis://localhost:6379")
    yield client
    await client.aclose()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_websocket_rate_limit(db_client, redis_client, client):
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
    
    # Simulate 5 WebSocket connections
    async def connect_websocket(index):
        ws = WebSocketTestSession(client, f"/monitoring/kpis/stream?token={access_token}&session_id={session_id}&user_id={user_id}")
        try:
            await ws.connect()
            await ws.receive_json()
            return True
        except Exception as e:
            return str(e)
    
    tasks = [connect_websocket(i) for i in range(6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify first 5 connections succeed, 6th fails
    success_count = sum(1 for r in results if r is True)
    assert success_count == 5
    assert any("WebSocket connection limit reached" in str(r) for r in results)
    
    # Verify Redis counter
    redis_key = f"websocket_limit:{user_id}"
    count = await redis_client.get(redis_key)
    assert int(count) == 5
    
    # Clean up
    await redis_client.delete(redis_key)

@pytest.mark.asyncio
async def test_kpi_streaming(db_client, client):
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
    
    async with client.websocket_connect(f"/monitoring/kpis/stream?token={access_token}&session_id={session_id}&user_id={user_id}") as ws:
        data = await ws.receive_json()
        assert data["type"] == "kpi_update"
        assert "data" in data
        assert "auth_success_rate" in data["data"]
        assert "auth_failure_count" in data["data"]
        assert "active_sessions" in data["data"]
        assert "anomalies_detected" in data["data"]
