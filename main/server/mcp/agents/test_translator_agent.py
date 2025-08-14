import pytest
from fastapi.testclient import TestClient
from main.server.mcp.agents.translator_agent import TranslatorAgent, TranslationRequest
from main.server.mcp.db.db_manager import DatabaseManager
from main.server.mcp.error_handler import ErrorHandler
from main.server.unified_server import app
import aiohttp

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def translator_agent():
    """Create a TranslatorAgent instance."""
    db_manager = DatabaseManager()
    error_handler = ErrorHandler()
    return TranslatorAgent(db_manager, error_handler)

@pytest.mark.asyncio
async def test_process_task_success(translator_agent, mocker):
    """Test successful translation task."""
    mocker.patch.object(translator_agent.base_prompt, 'generate_prompt', return_value="Translate the following text: Hola")
    mocker.patch('aiohttp.ClientSession.post', return_value=mocker.AsyncMock(status=200, json=mocker.AsyncMock(return_value={"translated_text": "Hello"})))
    mocker.patch.object(translator_agent.db_manager, 'log_translation', return_value=None)
    parameters = {"text": "Hola", "source_lang": "es", "target_lang": "en"}
    response = await translator_agent.process_task(parameters)
    assert response == {"translated_text": "Hello", "source_lang": "es", "target_lang": "en"}

@pytest.mark.asyncio
async def test_process_task_api_failure(translator_agent, mocker):
    """Test translation task with API failure."""
    mocker.patch.object(translator_agent.base_prompt, 'generate_prompt', return_value="Translate the following text: Hola")
    mocker.patch('aiohttp.ClientSession.post', return_value=mocker.AsyncMock(status=500, json=mocker.AsyncMock(return_value={})))
    parameters = {"text": "Hola", "source_lang": "es", "target_lang": "en"}
    with pytest.raises(HTTPException) as exc:
        await translator_agent.process_task(parameters)
    assert exc.value.status_code == 500
    assert "Translation API failed" in exc.value.detail
