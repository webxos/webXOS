# main/server/mcp/notes/test_mcp_server_notes.py
import pytest
from pymongo import MongoClient
from ..notes.mcp_server_notes import NotesService, MCPError

@pytest.fixture
def notes_service():
    service = NotesService()
    yield service
    service.collection.delete_many({})
    service.close()

@pytest.mark.asyncio
async def test_create_note(notes_service):
    result = await notes_service.create_note(
        title="Test Note",
        content="This is a test note",
        tags=["test", "mcp"],
        user_id="test_user"
    )
    assert result["status"] == "success"
    assert "note_id" in result

@pytest.mark.asyncio
async def test_create_note_invalid(notes_service):
    with pytest.raises(MCPError) as exc_info:
        await notes_service.create_note(
            title="",
            content="Invalid",
            tags=["test"],
            user_id="test_user"
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Title and content are required"

@pytest.mark.asyncio
async def test_get_note(notes_service):
    create_result = await notes_service.create_note(
        title="Test Note",
        content="This is a test note",
        tags=["test"],
        user_id="test_user"
    )
    note_id = create_result["note_id"]
    note = await notes_service.get_note(note_id, "test_user")
    assert note["title"] == "Test Note"
    assert note["content"] == "This is a test note"
    assert note["tags"] == ["test"]
    assert note["user_id"] == "test_user"

@pytest.mark.asyncio
async def test_get_note_not_found(notes_service):
    with pytest.raises(MCPError) as exc_info:
        await notes_service.get_note("invalid_id", "test_user")
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Note not found or access denied"

@pytest.mark.asyncio
async def test_search_notes(notes_service):
    await notes_service.create_note(
        title="Test Note 1",
        content="Content 1",
        tags=["test", "mcp"],
        user_id="test_user"
    )
    await notes_service.create_note(
        title="Test Note 2",
        content="Content 2",
        tags=["test"],
        user_id="test_user"
    )
    notes = await notes_service.search_notes(user_id="test_user", tags=["test"])
    assert len(notes) == 2
    assert all(note["user_id"] == "test_user" for note in notes)
    assert all("test" in note["tags"] for note in notes)
