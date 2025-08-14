import pytest
from fastapi.testclient import TestClient
from main.server.mcp.config_validator import ConfigValidator
from main.server.unified_server import app
import os

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def config_validator():
    """Create a ConfigValidator instance."""
    return ConfigValidator()

@pytest.mark.asyncio
async def test_validate_env_success(config_validator, mocker):
    """Test successful environment variable validation."""
    mocker.patch.dict(os.environ, {
        "POSTGRES_HOST": "postgresdb",
        "POSTGRES_DOCKER_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "vial_mcp",
        "MYSQL_HOST": "mysqldb",
        "MYSQL_DOCKER_PORT": "3306",
        "MYSQL_USER": "root",
        "MYSQL_ROOT_PASSWORD": "mysql",
        "MYSQL_DB": "vial_mcp",
        "MONGO_HOST": "mongodb",
        "MONGO_DOCKER_PORT": "27017",
        "MONGO_USER": "mongo",
        "MONGO_PASSWORD": "mongo",
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "JWT_SECRET": "your_jwt_secret",
        "PYTHON_LOCAL_PORT": "8000",
        "PYTHON_DOCKER_PORT": "8000",
        "NODE_LOCAL_PORT": "8080",
        "NODE_DOCKER_PORT": "8080",
        "JAVA_LOCAL_PORT": "8081",
        "JAVA_DOCKER_PORT": "8081"
    })
    config_validator.validate_env()  # Should not raise an exception

@pytest.mark.asyncio
async def test_validate_env_missing(config_validator, mocker):
    """Test environment validation with missing variable."""
    mocker.patch.dict(os.environ, {"POSTGRES_HOST": "postgresdb"})
    with pytest.raises(HTTPException) as exc:
        config_validator.validate_env()
    assert exc.value.status_code == 500
    assert "Missing environment variable" in exc.value.detail

@pytest.mark.asyncio
async def test_validate_env_invalid_port(config_validator, mocker):
    """Test environment validation with invalid port."""
    mocker.patch.dict(os.environ, {
        "POSTGRES_HOST": "postgresdb",
        "POSTGRES_DOCKER_PORT": "invalid",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "vial_mcp",
        "MYSQL_HOST": "mysqldb",
        "MYSQL_DOCKER_PORT": "3306",
        "MYSQL_USER": "root",
        "MYSQL_ROOT_PASSWORD": "mysql",
        "MYSQL_DB": "vial_mcp",
        "MONGO_HOST": "mongodb",
        "MONGO_DOCKER_PORT": "27017",
        "MONGO_USER": "mongo",
        "MONGO_PASSWORD": "mongo",
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "JWT_SECRET": "your_jwt_secret",
        "PYTHON_LOCAL_PORT": "8000",
        "PYTHON_DOCKER_PORT": "8000",
        "NODE_LOCAL_PORT": "8080",
        "NODE_DOCKER_PORT": "8080",
        "JAVA_LOCAL_PORT": "8081",
        "JAVA_DOCKER_PORT": "8081"
    })
    with pytest.raises(HTTPException) as exc:
        config_validator.validate_env()
    assert exc.value.status_code == 500
    assert "Invalid port value" in exc.value.detail

@pytest.mark.asyncio
async def test_validate_db_configs_success(config_validator):
    """Test successful database configuration validation."""
    postgres_config = {
        "host": "postgresdb", "port": 5432, "user": "postgres", "password": "postgres", "database": "vial_mcp"
    }
    mysql_config = {
        "host": "mysqldb", "port": 3306, "user": "root", "password": "mysql", "database": "vial_mcp"
    }
    mongo_config = {
        "host": "mongodb", "port": 27017, "username": "mongo", "password": "mongo"
    }
    config_validator.validate_db_configs(postgres_config, mysql_config, mongo_config)  # Should not raise an exception

@pytest.mark.asyncio
async def test_validate_db_configs_missing(config_validator):
    """Test database configuration validation with missing field."""
    postgres_config = {
        "host": "postgresdb", "port": 5432, "user": "postgres", "database": "vial_mcp"  # Missing password
    }
    mysql_config = {
        "host": "mysqldb", "port": 3306, "user": "root", "password": "mysql", "database": "vial_mcp"
    }
    mongo_config = {
        "host": "mongodb", "port": 27017, "username": "mongo", "password": "mongo"
    }
    with pytest.raises(HTTPException) as exc:
        config_validator.validate_db_configs(postgres_config, mysql_config, mongo_config)
    assert exc.value.status_code == 500
    assert "Invalid PostgreSQL config: missing or empty password" in exc.value.detail
