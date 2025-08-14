import pytest
import asyncpg
import mysql.connector
from pymongo import MongoClient
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.unified_server import app

@pytest.fixture
def db_manager():
    """Create a DatabaseManager instance with mock configs."""
    postgres_config = {
        "host": "postgresdb",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": "vial_mcp"
    }
    mysql_config = {
        "host": "mysqldb",
        "port": 3306,
        "user": "root",
        "password": "mysql",
        "database": "vial_mcp"
    }
    mongo_config = {
        "host": "mongodb",
        "port": 27017,
        "username": "mongo",
        "password": "mongo",
        "database": "vial_mcp"
    }
    return DatabaseManager(postgres_config, mysql_config, mongo_config)

@pytest.mark.asyncio
async def test_connect_success(db_manager, mocker):
    """Test successful database connections."""
    mocker.patch.object(asyncpg, 'create_pool', return_value=mocker.AsyncMock())
    mocker.patch.object(mysql.connector, 'connect', return_value=mocker.MagicMock())
    mocker.patch.object(MongoClient, '__init__', return_value=None)
    await db_manager.connect()
    assert db_manager.postgres_pool is not None
    assert db_manager.mysql_conn is not None
    assert db_manager.mongo_client is not None

@pytest.mark.asyncio
async def test_add_note_postgres(db_manager, mocker):
    """Test adding a note to PostgreSQL."""
    mocker.patch.object(db_manager, 'postgres_pool', mocker.AsyncMock())
    mock_conn = mocker.AsyncMock()
    mocker.patch.object(db_manager.postgres_pool, 'acquire', return_value=mock_conn)
    response = await db_manager.add_note("wallet_123", "Test note", "res_123", "postgres")
    assert "note_id" in response
    assert response["wallet_id"] == "wallet_123"
    assert response["content"] == "Test note"
    assert response["resource_id"] == "res_123"
    assert response["db_type"] == "postgres"

@pytest.mark.asyncio
async def test_add_note_mysql(db_manager, mocker):
    """Test adding a note to MySQL."""
    mocker.patch.object(db_manager, 'mysql_conn', mocker.MagicMock())
    mock_cursor = mocker.MagicMock()
    mocker.patch.object(db_manager.mysql_conn, 'cursor', return_value=mock_cursor)
    response = await db_manager.add_note("wallet_123", "Test note", "res_123", "mysql")
    assert "note_id" in response
    assert response["wallet_id"] == "wallet_123"
    assert response["content"] == "Test note"
    assert response["resource_id"] == "res_123"
    assert response["db_type"] == "mysql"

@pytest.mark.asyncio
async def test_add_note_mongo(db_manager, mocker):
    """Test adding a note to MongoDB."""
    mocker.patch.object(db_manager, 'mongo_client', mocker.MagicMock())
    mock_db = mocker.MagicMock()
    mocker.patch.object(db_manager.mongo_client, '__getitem__', return_value=mock_db)
    response = await db_manager.add_note("wallet_123", "Test note", "res_123", "mongo")
    assert "note_id" in response
    assert response["wallet_id"] == "wallet_123"
    assert response["content"] == "Test note"
    assert response["resource_id"] == "res_123"
    assert response["db_type"] == "mongo"

@pytest.mark.asyncio
async def test_get_notes_postgres(db_manager, mocker):
    """Test retrieving notes from PostgreSQL."""
    mocker.patch.object(db_manager, 'postgres_pool', mocker.AsyncMock())
    mock_conn = mocker.AsyncMock()
    mocker.patch.object(db_manager.postgres_pool, 'acquire', return_value=mock_conn)
    mock_conn.fetch.return_value = [{"note_id": "note_123", "content": "Test note"}]
    notes = await db_manager.get_notes("wallet_123", 10, "postgres")
    assert len(notes) == 1
    assert notes[0]["note_id"] == "note_123"
    assert notes[0]["content"] == "Test note"

@pytest.mark.asyncio
async def test_get_notes_mysql(db_manager, mocker):
    """Test retrieving notes from MySQL."""
    mocker.patch.object(db_manager, 'mysql_conn', mocker.MagicMock())
    mock_cursor = mocker.MagicMock()
    mocker.patch.object(db_manager.mysql_conn, 'cursor', return_value=mock_cursor)
    mock_cursor.fetchall.return_value = [{"note_id": "note_123", "content": "Test note"}]
    notes = await db_manager.get_notes("wallet_123", 10, "mysql")
    assert len(notes) == 1
    assert notes[0]["note_id"] == "note_123"
    assert notes[0]["content"] == "Test note"

@pytest.mark.asyncio
async def test_get_notes_mongo(db_manager, mocker):
    """Test retrieving notes from MongoDB."""
    mocker.patch.object(db_manager, 'mongo_client', mocker.MagicMock())
    mock_db = mocker.MagicMock()
    mocker.patch.object(db_manager.mongo_client, '__getitem__', return_value=mock_db)
    mock_db.notes.find.return_value.sort.return_value.limit.return_value.to_list.return_value = [{"note_id": "note_123", "content": "Test note"}]
    notes = await db_manager.get_notes("wallet_123", 10, "mongo")
    assert len(notes) == 1
    assert notes[0]["note_id"] == "note_123"
    assert notes[0]["content"] == "Test note"
