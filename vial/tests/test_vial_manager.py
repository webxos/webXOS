import pytest
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet

def test_vial_manager_initialization():
    wallet = WebXOSWallet()
    manager = VialManager(wallet)
    assert len(manager.vials) == 4
    assert all(vial["status"] == "stopped" for vial in manager.vials.values())
    assert all(vial["wallet"]["address"] in wallet.addresses for vial in manager.vials.values())

def test_train_vials():
    wallet = WebXOSWallet()
    manager = VialManager(wallet)
    balance = manager.train_vials("test-network", "print('test')", "test.py")
    assert balance == 0.0004
    assert all(vial["status"] == "running" for vial in manager.vials.values())
    assert all(vial["code_length"] == len("print('test')") for vial in manager.vials.values())

def test_reset_vials():
    wallet = WebXOSWallet()
    manager = VialManager(wallet)
    manager.train_vials("test-network", "print('test')", "test.py")
    manager.reset_vials()
    assert all(vial["status"] == "stopped" for vial in manager.vials.values())
    assert all(vial["code_length"] == 0 for vial in manager.vials.values())
    assert all(vial["wallet"]["balance"] == 0.0 for vial in manager.vials.values())
