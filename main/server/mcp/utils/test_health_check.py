# main/server/mcp/utils/test_health_check.py
import pytest
from fastapi.testclient import TestClient
from ..utils.health_check import HealthStatus, router
from ..utils.mcp_error_handler import MCPError

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
def health_status(mocker):
    status = HealthStatus()
    yield status
    status.close()

@pytest.mark.asyncio
async def test_health_check_healthy(client, health_status, mocker):
    mocker.patch.object(health_status, 'check_mongodb', return_value=True)
    mocker.patch.object(health_status, 'check_redis', return_value=True)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "services": {
            "mongodb": "up",
            "redis": "up"
        }
    }

@pytest.mark.asyncio
async def test_health_check_unhealthy(client, health_status, mocker):
    mocker.patch.object(health_status, 'check_mongodb', return_value=False)
    mocker.patch.object(health_status, 'check_redis', return_value=True)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "unhealthy",
        "services": {
            "mongodb": "down",
            "redis": "up"
        }
    }

@pytest.mark.asyncio
async def test_readiness_check_ready(client, health_status, mocker):
    mocker.patch.object(health_status, 'check_mongodb', return_value=True)
    mocker.patch.object(health_status, 'check_redis', return_value=True)
    
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "services": {
            "mongodb": "up",
            "redis": "up"
        }
    }

@pytest.mark.asyncio
async def test_readiness_check_not_ready(client, health_status, mocker):
    mocker.patch.object(health_status, 'check_mongodb', return_value=False)
    mocker.patch.object(health_status, 'check_redis', return_value=True)
    
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "not ready",
        "services": {
            "mongodb": "down",
            "redis": "up"
        }
    }
