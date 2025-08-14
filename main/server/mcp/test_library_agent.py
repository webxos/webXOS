import pytest
from fastapi.testclient import TestClient
from main.server.mcp.library_agent import LibraryAgent
from main.server.mcp.auth_manager import AuthManager
from main.server.unified_server import app
import psycopg2
import pymongo

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def library_agent():
    """Create a LibraryAgent instance."""
    return LibraryAgent()

@pytest.mark.asyncio
async def test_process_library_success(library_agent, mocker):
    """Test successful library processing."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch("psycopg2.connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(fetchone=lambda: ["lib_123"], execute=lambda *args, **kwargs: None),
        commit=lambda: None
    ))
    response = await library_agent.process_library("lib_123", "wallet_123", "test_content", "postgres", "test_token")
    assert response["status"] == "success"
    assert response["library_id"] == "lib_123"
    assert "vector_id" in response

@pytest.mark.asyncio
async def test_process_library_unauthorized(library_agent, mocker):
    """Test library processing with unauthorized wallet."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_456"})
    with pytest.raises(HTTPException) as exc:
        await library_agent.process_library("lib_123", "wallet_123", "test_content", "postgres", "test_token")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Unauthorized wallet access"

@pytest.mark.asyncio
async def test_process_library_mongo(library_agent, mocker):
    """Test library processing with MongoDB."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(pymongo.collection.Collection, 'insert_one', return_value=mocker.MagicMock(inserted_id="lib_123"))
    response = await library_agent.process_library("lib_123", "wallet_123", "test_content", "mongo", "test_token")
    assert response["status"] == "success"
    assert response["library_id"] == "lib_123"
    assert "vector_id" in response

@pytest.mark.asyncio
async def test_process_library_invalid_db(library_agent, mocker):
    """Test library processing with invalid database type."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_123"})
    with pytest.raises(ValueError) as exc:
        await library_agent.process_library("lib_123", "wallet_123", "test_content", "invalid", "test_token")
    assert str(exc.value) == "Invalid database type"
