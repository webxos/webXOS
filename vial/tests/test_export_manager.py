import pytest
from export_manager import ExportManager
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet

def test_export_to_markdown():
    wallet = WebXOSWallet()
    vial_manager = VialManager(wallet)
    export_manager = ExportManager(vial_manager, wallet)
    wallet.update_balance("test_network", 0.1)
    markdown = export_manager.export_to_markdown("test_token", "test_network")
    assert "Vial MCP Export" in markdown
    assert "Token: test_token" in markdown
    assert "Network ID: test_network" in markdown
    assert "Wallet Data" in markdown
    assert "Balance: 0.1 $WEBXOS" in markdown