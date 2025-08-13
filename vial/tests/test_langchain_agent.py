import pytest
from vial.langchain_agent import LangChainAgent
from unittest.mock import patch

@pytest.fixture
def langchain_agent():
    return LangChainAgent()

def test_call_llm_success(langchain_agent):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "mocked response"}
        mock_post.return_value.status_code = 200
        result = langchain_agent.call_llm("test prompt", "user123", "llama3.3")
        assert result["status"] == "success"
        assert result["response"] == "mocked response"

def test_call_llm_invalid_model(langchain_agent):
    with pytest.raises(ValueError) as exc:
        langchain_agent.call_llm("test prompt", "user123", "invalid_model")
    assert "Invalid model" in str(exc.value)

def test_call_llm_api_error(langchain_agent):
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {"error": "Invalid API key"}
        with pytest.raises(Exception) as exc:
            langchain_agent.call_llm("test prompt", "user123", "llama3.3")
        assert "Invalid API key" in str(exc.value)

def test_call_llm_logging(langchain_agent, tmp_path):
    error_log = tmp_path / "errorlog.md"
    with open(error_log, "a") as f:
        f.write("")
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"response": "mocked response"}
        mock_post.return_value.status_code = 200
        langchain_agent.call_llm("test prompt", "user123", "llama3.3")
        with open(error_log) as f:
            log_content = f.read()
        assert "LLM call by user123" in log_content
