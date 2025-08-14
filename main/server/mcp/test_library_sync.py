import pytest
from fastapi.testclient import TestClient
from main.server.mcp.library_sync import LibrarySync
from main.server.unified_server import app
import psycopg2
import pymongo

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def library_sync():
    """Create a LibrarySync instance."""
    return LibrarySync()

@pytest.mark.asyncio
async def test_sync_library_success(library_sync, mocker):
    """Test successful library synchronization."""
    mocker.patch.object(pymongo.collection.Collection, 'find_one', return_value={
        "library_id": "lib_123", "wallet_id": "wallet_123", "content": "test_content", "timestamp": datetime.now()
    })
    mocker.patch("psycopg2.connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(execute=lambda *args, **kwargs: None),
        commit=lambda: None
    ))
    mocker.patch("mysql.connector.connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(execute=lambda *args, **kwargs: None),
        commit=lambda: None
    ))
    response = await library_sync.sync_library("lib_123", "wallet_123")
    assert response["status"] == "success"
    assert response["library_id"] == "lib_123"
    assert response["wallet_id"] == "wallet_123"

@pytest.mark.asyncio
async def test_sync_library_not_found(library_sync, mocker):
    """Test library synchronization with non-existent library."""
    mocker.patch.object(pymongo.collection.Collection, 'find_one', return_value=None)
    with pytest.raises(HTTPException) as exc:
        await library_sync.sync_library("lib_123", "wallet_123")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Library not found"

@pytest.mark.asyncio
async def test_sync_library_postgres_failure(library_sync, mocker):
    """Test library synchronization with PostgreSQL failure."""
    mocker.patch.object(pymongo.collection.Collection, 'find_one', return_value={
        "library_id": "lib_123", "wallet_id": "wallet_123", "content": "test_content", "timestamp": datetime.now()
    })
    mocker.patch("psycopg2.connect", side_effect=Exception("Database error"))
    with pytest.raises(HTTPException) as exc:
        await library_sync.sync_library("lib_123", "wallet_123")
    assert exc.value.status_code == 500
    assert "Library sync failed" in exc.value.detail
