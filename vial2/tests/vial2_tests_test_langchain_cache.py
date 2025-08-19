import pytest
from vial2.mcp.langchain import mcp_chain

def test_cache_hit():
    chain = mcp_chain.MCPChain()
    chain.cache["test"] = "cached"
    assert chain.get_from_cache("test") == "cached"

def test_cache_miss():
    chain = mcp_chain.MCPChain()
    assert chain.get_from_cache("missing") is None