import pytest
from vial.langchain_agent import LangChainAgent
from vial.webxos_wallet import WebXOSWallet

@pytest.fixture
def langchain_agent():
    return LangChainAgent()

@pytest.fixture
def wallet():
    wallet = WebXOSWallet()
    wallet.conn.execute("DELETE FROM balances")
    wallet.conn.commit()
    return wallet

def test_enhance_query(langchain_agent):
    query = "test query"
    enhanced = langchain_agent.enhance_query(query)
    assert enhanced == "test query (enhanced)"  # Placeholder LLM behavior

def test_train_vial(langchain_agent, wallet):
    wallet.update_balance("vial1", 1.0)
    assert langchain_agent.train_vial("vial1", "test query", 1.0)
    assert wallet.get_balance("vial1") == 0.9999
    with pytest.raises(ValueError):
        langchain_agent.train_vial("vial1", "test query", 0.0)
