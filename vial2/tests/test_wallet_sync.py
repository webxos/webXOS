import pytest
from vial2.mcp.api import wallet_sync

def test_sync_wallet():
    result = wallet_sync.sync_wallet({"address": "test_address"})
    assert result["status"] == "synced"
