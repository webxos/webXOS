import pytest
from fastapi.testclient import TestClient
from main.server.mcp.notes.mcp_server_notes import MCPNotesHandler, NoteRequest, NoteReadRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.cache_manager import CacheManager
from main.server.mcp.security_manager import SecurityManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def notes_handler():
    """Create an MCPNotesHandler instance."""
    db_manager = DatabaseManager()
    cache_manager = CacheManager()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    return MCPNotesHandler(db_manager, cache_manager, security_manager, error_handler)

@pytest.mark.asyncio
async def test_add_note_success(notes_handler, mocker):
    """Test successful note addition."""
    mocker.patch.object(notes_handler.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(notes_handler.db_manager, 'add_note', return_value={"note_id": "note_123", "wallet_id": "wallet_123", "content": "Test note", "resource_id": "res_123", "db_type": "postgres"})
    mocker.patch.object(notes_handler.cache_manager, 'invalidate_cache')
    request = NoteRequest(wallet_id="wallet_123", content="Test note", resource_id="res_123", db_type="postgres")
    response = await notes_handler.add_note(request, "mocked_token")
    assert response == {"note_id": "note_123", "wallet_id": "wallet_123", "content": "Test note", "resource_id": "res_123", "db_type": "postgres"}

@pytest.mark.asyncio
async def test_add_note_unauthorized(notes_handler, mocker):
    """Test note addition with unauthorized wallet."""
    mocker.patch.object(notes_handler.security_manager, 'validate_token', return_value={"wallet_id": "wrong_wallet"})
    request = NoteRequest(wallet_id="wallet_123", content="Test note", resource_id="res_123", db_type="postgres")
    with pytest.raises(HTTPException) as exc:
        await notes_handler.add_note(request, "mocked_token")
    assert exc.value.status_code == 500
    assert "Unauthorized wallet access" in exc.value.detail

@pytest.mark.asyncio
async def test_read_note_success(notes_handler, mocker):
    """Test successful note retrieval."""
    mocker.patch.object(notes_handler.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(notes_handler.cache_manager, 'get_cached_response', return_value=None)
    mocker.patch.object(notes_handler.db_manager, 'get_notes', return_value=[{"note_id": "note_123", "content": "Test note"}])
    mocker.patch.object(notes_handler.cache_manager, 'cache_response')
    request = NoteReadRequest(wallet_id="wallet_123", db_type="postgres")
    response = await notes_handler.read_note(request, "mocked_token")
    assert response == {"notes": [{"note_id": "note_123", "content": "Test note"}]}

@pytest.mark.asyncio
async def test_read_note_cached(notes_handler, mocker):
    """Test note retrieval from cache."""
    mocker.patch.object(notes_handler.security_manager, 'validate_token', return_value={"wallet_id": "wallet_123"})
    mocker.patch.object(notes_handler.cache_manager, 'get_cached_response', return_value=[{"note_id": "note_123", "content": "Test note"}])
    request = NoteReadRequest(wallet_id="wallet_123", db_type="postgres")
    response = await notes_handler.read_note(request, "mocked_token")
    assert response == {"notes": [{"note_id": "note_123", "content": "Test note"}]}
