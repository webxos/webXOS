import pytest
from webxos_wallet import WebXOSWallet

def test_wallet_update():
    wallet = WebXOSWallet()
    balance = wallet.update_balance("test_network", 0.1)
    assert balance == 0.1
    assert wallet.get_balance("test_network") == 0.1

def test_wallet_export():
    wallet = WebXOSWallet()
    wallet.update_balance("test_network", 0.1)
    markdown = wallet.export_wallet("test_network")
    assert "Network ID: test_network" in markdown
    assert "Balance: 0.1 $WEBXOS" in markdown
    assert "Transactions" in markdown