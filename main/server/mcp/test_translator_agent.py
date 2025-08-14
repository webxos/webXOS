import pytest
from fastapi.testclient import TestClient
from main.server.mcp.translator_agent import TranslatorAgent
from main.server.mcp.auth_manager import AuthManager
from main.server.unified_server import app

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def translator_agent():
    """Create a TranslatorAgent instance."""
    return TranslatorAgent()

@pytest.mark.asyncio
async def test_translate_content_success(translator_agent, mocker):
    """Test successful content translation."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_123"})
    response = await translator_agent.translate_content("Hello", "es", "wallet_123", "test_token")
    assert response["original_content"] == "Hello"
    assert response["target_language"] == "es"
    assert "translated_content" in response
    assert "timestamp" in response

@pytest.mark.asyncio
async def test_translate_content_unauthorized(translator_agent, mocker):
    """Test translation with unauthorized wallet."""
    mocker.patch.object(AuthManager, 'verify_token', return_value={"wallet_id": "wallet_456"})
    with pytest.raises(HTTPException) as exc:
        await translator_agent.translate_content("Hello", "es", "wallet_123", "test_token")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Unauthorized wallet access"
