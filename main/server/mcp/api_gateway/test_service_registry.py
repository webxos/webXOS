# main/server/mcp/api_gateway/test_service_registry.py
import pytest
from pymongo import MongoClient
from ..api_gateway.service_registry import ServiceRegistry, MCPError

@pytest.fixture
async def service_registry():
    registry = ServiceRegistry()
    yield registry
    registry.collection.delete_many({})
    registry.close()

@pytest.mark.asyncio
async def test_register_service(service_registry):
    service_id = await service_registry.register_service(
        service_name="notes_service",
        address="localhost",
        port=8081,
        metadata={"version": "1.0"}
    )
    assert service_id == "notes_service:localhost:8081"
    services = await service_registry.get_services("notes_service")
    assert len(services) == 1
    assert services[0]["address"] == "localhost"
    assert services[0]["port"] == 8081

@pytest.mark.asyncio
async def test_register_service_invalid(service_registry):
    with pytest.raises(MCPError) as exc_info:
        await service_registry.register_service(
            service_name="",
            address="localhost",
            port=8081,
            metadata={"version": "1.0"}
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Service name, address, and port are required"

@pytest.mark.asyncio
async def test_deregister_service(service_registry):
    service_id = await service_registry.register_service(
        service_name="notes_service",
        address="localhost",
        port=8081,
        metadata={"version": "1.0"}
    )
    await service_registry.deregister_service(service_id)
    services = await service_registry.get_services("notes_service")
    assert len(services) == 0

@pytest.mark.asyncio
async def test_deregister_service_not_found(service_registry):
    with pytest.raises(MCPError) as exc_info:
        await service_registry.deregister_service("invalid_id")
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Service not found"

@pytest.mark.asyncio
async def test_update_heartbeat(service_registry):
    service_id = await service_registry.register_service(
        service_name="notes_service",
        address="localhost",
        port=8081,
        metadata={"version": "1.0"}
    )
    await service_registry.update_heartbeat(service_id)
    services = await service_registry.get_services("notes_service")
    assert len(services) == 1
