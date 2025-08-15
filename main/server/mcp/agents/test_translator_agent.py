# main/server/mcp/agents/test_translator_agent.py
import pytest
from pymongo import MongoClient
from ..agents.translator_agent import TranslatorAgent, MCPError
from ..agents.global_mcp_agents import GlobalMCPAgents

@pytest.fixture
def translator_agent():
    agent = TranslatorAgent()
    yield agent
    agent.collection.delete_many({})
    agent.close()

@pytest.fixture
def global_agents():
    agents = GlobalMCPAgents()
    yield agents
    agents.collection.delete_many({})
    agents.close()

@pytest.mark.asyncio
async def test_translate_text(translator_agent, global_agents, mocker):
    # Mock translation API
    mocker.patch("requests.post", return_value=mocker.Mock(status_code=200, json=lambda: {"translated_text": "Hola"}))
    
    # Create agent
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["translate"],
        config={"lang": "es"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]

    # Test translation
    result = await translator_agent.translate_text(
        agent_id=agent_id,
        text="Hello",
        source_lang="en",
        target_lang="es",
        user_id="test_user"
    )
    assert result["status"] == "success"
    assert result["translated_text"] == "Hola"
    assert "translation_id" in result

@pytest.mark.asyncio
async def test_translate_text_invalid_agent(translator_agent):
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate_text(
            agent_id="invalid_id",
            text="Hello",
            source_lang="en",
            target_lang="es",
            user_id="test_user"
        )
    assert exc_info.value.code == -32003
    assert exc_info.value.message == "Agent not found or access denied"

@pytest.mark.asyncio
async def test_translate_text_invalid_lang(translator_agent, global_agents):
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["translate"],
        config={"lang": "es"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    with pytest.raises(MCPError) as exc_info:
        await translator_agent.translate_text(
            agent_id=agent_id,
            text="Hello",
            source_lang="xx",
            target_lang="es",
            user_id="test_user"
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Unsupported language code"

@pytest.mark.asyncio
async def test_get_translation_history(translator_agent, global_agents, mocker):
    mocker.patch("requests.post", return_value=mocker.Mock(status_code=200, json=lambda: {"translated_text": "Hola"}))
    
    create_result = await global_agents.create_agent(
        vial_id="vial1",
        tasks=["translate"],
        config={"lang": "es"},
        user_id="test_user"
    )
    agent_id = create_result["agent_id"]
    
    await translator_agent.translate_text(
        agent_id=agent_id,
        text="Hello",
        source_lang="en",
        target_lang="es",
        user_id="test_user"
    )
    
    history = await translator_agent.get_translation_history(agent_id, "test_user")
    assert len(history) == 1
    assert history[0]["source_text"] == "Hello"
    assert history[0]["translated_text"] == "Hola"
    assert history[0]["source_lang"] == "en"
    assert history[0]["target_lang"] == "es"
