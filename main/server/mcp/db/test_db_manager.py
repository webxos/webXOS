import pytest
import psycopg2
import mysql.connector
from pymongo import MongoClient
from fastapi.testclient import TestClient
from main.server.unified_server import app
from main.server.mcp.db.db_manager import DatabaseManager

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def db_manager():
    """Create a DatabaseManager instance with test configurations."""
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
        "password": "mongo"
    }
    return DatabaseManager(postgres_config, mysql_config, mongo_config)

@pytest.mark.asyncio
async def test_add_note_postgres(db_manager, mocker):
    """Test adding a note to PostgreSQL."""
    mocker.patch.object(psycopg2, "connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(
            fetchone=lambda: [1],
            execute=lambda *args, **kwargs: None
        ),
        commit=lambda: None
    ))
    result = db_manager.add_note("wallet_123", "Test note", db_type="postgres")
    assert result["status"] == "success"
    assert result["note_id"] == 1
    assert result["wallet_id"] == "wallet_123"

@pytest.mark.asyncio
async def test_add_note_mysql(db_manager, mocker):
    """Test adding a note to MySQL."""
    mocker.patch.object(mysql.connector, "connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(
            lastrowid=1,
            execute=lambda *args, **kwargs: None
        ),
        commit=lambda: None
    ))
    result = db_manager.add_note("wallet_123", "Test note", db_type="mysql")
    assert result["status"] == "success"
    assert result["note_id"] == 1
    assert result["wallet_id"] == "wallet_123"

@pytest.mark.asyncio
async def test_add_note_mongo(db_manager, mocker):
    """Test adding a note to MongoDB."""
    mocker.patch.object(MongoClient, "__init__", return_value=None)
    mocker.patch.object(MongoClient, "vial_mcp", create=True)
    mocker.patch("pymongo.collection.Collection.insert_one", return_value=mocker.MagicMock(inserted_id="123"))
    result = db_manager.add_note("wallet_123", "Test note", db_type="mongo")
    assert result["status"] == "success"
    assert result["note_id"] == "123"
    assert result["wallet_id"] == "wallet_123"

@pytest.mark.asyncio
async def test_get_notes_postgres(db_manager, mocker):
    """Test retrieving notes from PostgreSQL."""
    mocker.patch.object(psycopg2, "connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(
            fetchall=lambda: [(1, "Test note", None, datetime.now(), "wallet_123")]
        )
    ))
    result = db_manager.get_notes("wallet_123", 10, "postgres")
    assert result["status"] == "success"
    assert len(result["notes"]) == 1
    assert result["notes"][0]["content"] == "Test note"

@pytest.mark.asyncio
async def test_get_notes_mysql(db_manager, mocker):
    """Test retrieving notes from MySQL."""
    mocker.patch.object(mysql.connector, "connect", return_value=mocker.MagicMock(
        cursor=mocker.MagicMock(
            fetchall=lambda: [(1, "Test note", None, datetime.now(), "wallet_123")]
        )
    ))
    result = db_manager.get_notes("wallet_123", 10, "mysql")
    assert result["status"] == "success"
    assert len(result["notes"]) == 1
    assert result["notes"][0]["content"] == "Test note"

@pytest.mark.asyncio
async def test_get_notes_mongo(db_manager, mocker):
    """Test retrieving notes from MongoDB."""
    mocker.patch.object(MongoClient, "__init__", return_value=None)
    mocker.patch.object(MongoClient, "vial_mcp", create=True)
    mocker.patch("pymongo.collection.Collection.find", return_value=mocker.MagicMock(
        sort=lambda *args: mocker.MagicMock(
            limit=lambda *args: [{
                "_id": "123",
                "content": "Test note",
                "resource_id": None,
                "timestamp": datetime.now(),
                "wallet_id": "wallet_123"
            }]
        )
    ))
    result = db_manager.get_notes("wallet_123", 10, "mongo")
    assert result["status"] == "success"
    assert len(result["notes"]) == 1
    assert result["notes"][0]["content"] == "Test note"
