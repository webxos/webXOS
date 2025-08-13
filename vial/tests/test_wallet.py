import pytest
from vial.webxos_wallet import WebXOSWallet
import sqlite3
import os

@pytest.fixture
def wallet():
    wallet = WebXOSWallet()
    wallet.conn.execute("DELETE FROM balances")
    wallet.conn.execute("DELETE FROM transactions")
    wallet.conn.commit()
    return wallet

def test_update_balance(wallet):
    wallet.update_balance("test_network", 100.0)
    assert wallet.get_balance("test_network") == 100.0

def test_cashout(wallet):
    wallet.update_balance("test_network", 100.0)
    assert wallet.cashout("test_network", "test_address", 50.0)
    assert wallet.get_balance("test_network") == 50.0
    with pytest.raises(ValueError):
        wallet.cashout("test_network", "test_address", 100.0)

def test_validate_export(wallet):
    export_data = """# WebXOS Vial and Wallet Export
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
    assert wallet.validate_export(export_data)
    invalid_data = """# Invalid Export
## Wallet
- Hash: invalid_hash
"""
    assert not wallet.validate_export(invalid_data)
