import pytest
from vial2.mcp.langchain import mcp_chain

def test_store_vector():
    chain = mcp_chain.MCPChain()
    chain.store_vector("test", [1, 2, 3])
    assert chain.get_vector("test") == [1, 2, 3]