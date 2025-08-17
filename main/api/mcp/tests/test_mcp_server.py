import pytest
import asyncio
from mcp.client import create_stdio_client
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

@pytest.mark.asyncio
async def test_server_initialization():
    async with create_stdio_client() as client:
        result = await client.initialize()
        assert result.serverInfo.name == "vial-mcp"
        assert result.serverInfo.version == "3.0.0"
        assert result.serverInfo.description == "Vial MCP server for AI agent management"

@pytest.mark.asyncio
async def test_list_tools():
    async with create_stdio_client() as client:
        await client.initialize()
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools.tools]
        assert "authentication" in tool_names
        assert "wallet" in tool_names
        assert "blockchain" in tool_names
        assert "health" in tool_names
        assert "security" in tool_names
        assert "notifications" in tool_names

@pytest.mark.asyncio
async def test_call_auth_tool(db_client):
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
    
    async with create_stdio_client() as client:
        await client.initialize()
        result = await client.call_tool("authentication", {
            "method": "verifyToken",
            "user_id": user_id,
            "token": access_token,
            "session_id": session_id
        })
        assert result["user_id"] == user_id

@pytest.mark.asyncio
async def test_call_wallet_tool(db_client):
    user_id = str(uuid.uuid4())
    wallet_address = str(uuid.uuid4())
    
    # Setup test data
    await db_client.query(
        "INSERT INTO users (user_id, balance, wallet_address) VALUES ($1, $2, $3)",
        [user_id, 100.0, wallet_address]
    )
    
    async with create_stdio_client() as client:
        await client.initialize()
        result = await client.call_tool("wallet", {
            "method": "getVialBalance",
            "user_id": user_id,
            "vial_id": "vial1"
        })
        assert result["balance"] == 0.0
