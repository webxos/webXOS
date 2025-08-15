# main/server/mcp/api_gateway/test_service_registry.py
import pytest
from ..api_gateway.service_registry import ServiceRegistry, MCPError
from ..utils.performance_metrics import PerformanceMetrics
import asyncio

@pytest.fixture
async def service_registry():
    registry = ServiceRegistry()
    yield registry

@pytest.mark.asyncio
async def test_register_service(service_registry, mocker):
    mocker.patch("main.server.mcp.utils.api_config.APIConfig.validate_endpoint", return_value=True)
    async def mock_handler(params):
        return {"result": "success"}
    
    service_registry.register_service("mcp.testMethod", mock_handler)
    assert "mcp.testMethod" in service_registry.services
    assert service_registry.metrics.requests_total.labels(endpoint="mcp.testMethod")._value.get() == 0

@pytest.mark.asyncio
async def test_register_service_invalid_endpoint(service_registry, mocker):
    mocker.patch("main.server.mcp.utils.api_config.APIConfig.validate_endpoint", return_value=False)
    async def mock_handler(params):
        return {"result": "success"}
    
    with pytest.raises(MCPError) as exc_info:
        service_registry.register_service("mcp.invalidMethod", mock_handler)
    assert exc_info.value.code == -32601
    assert exc_info.value.message == "Method mcp.invalidMethod is not enabled"

@pytest.mark.asyncio
async def test_dispatch_success(service_registry, mocker):
    mocker.patch("main.server.mcp.utils.api_config.APIConfig.validate_endpoint", return_value=True)
    async def mock_handler(params):
        return {"result": "success"}
    
    service_registry.register_service("mcp.testMethod", mock_handler)
    response = await service_registry.dispatch("mcp.testMethod", {"param": "value"}, 1)
    assert response["jsonrpc"] == "2.0"
    assert response["result"] == {"result": "success"}
    assert response["id"] == 1
    assert service_registry.metrics.requests_total.labels(endpoint="mcp.testMethod")._value.get() == 1

@pytest.mark.asyncio
async def test_dispatch_method_not_found(service_registry):
    response = await service_registry.dispatch("mcp.unknownMethod", {"param": "value"}, 1)
    assert response["jsonrpc"] == "2.0"
    assert response["error"]["code"] == -32601
    assert response["error"]["message"] == "Method mcp.unknownMethod not found"
