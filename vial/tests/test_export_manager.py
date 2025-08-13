import pytest
from vial.export_manager import ExportManager
from vial.webxos_wallet import WebXOSWallet

@pytest.fixture
def export_manager():
    return ExportManager()

@pytest.fixture
def wallet():
    wallet = WebXOSWallet()
    wallet.conn.execute("DELETE FROM balances")
    wallet.conn.commit()
    return wallet

def test_generate_export(export_manager, wallet):
    wallet.update_balance("test_user", 100.0)
    vials = {
        "vial1": {"status": "running", "script": "print('test')", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial2": {"status": "running", "script": "print('test')", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial3": {"status": "running", "script": "print('test')", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial4": {"status": "running", "script": "print('test')", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"}
    }
    export_data = export_manager.generate_export("test_user", vials)
    assert "# WebXOS Vial and Wallet Export" in export_data
    assert "Wallet Balance: 100.0000 $WEBXOS" in export_data
    assert len(export_data.split("# Vial Agent: ")) == 5  # Header + 4 vials
    assert export_manager.validate_export(export_data)

def test_validate_export(export_manager):
    valid_export = """# WebXOS Vial and Wallet Export
## Wallet
- Hash: bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95
# Vial Agent: vial1
- Wallet Hash: bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95
# Vial Agent: vial2
- Wallet Hash: bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95
# Vial Agent: vial3
- Wallet Hash: bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95
# Vial Agent: vial4
- Wallet Hash: bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95
"""
    assert export_manager.validate_export(valid_export)
    invalid_export = """# Invalid Export
## Wallet
- Hash: invalid_hash
"""
    assert not export_manager.validate_export(invalid_export)
