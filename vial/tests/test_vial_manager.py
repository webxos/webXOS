import pytest
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet

def test_vial_manager_train():
    wallet = WebXOSWallet()
    manager = VialManager(wallet)
    balance = manager.train_vials("test_network", "sample content", "test.txt")
    assert balance > 0
    assert len(manager.get_vials()) == 4
    assert manager.network_state["test_network"]["last_trained"] == "test.txt"

def test_vial_manager_reset():
    wallet = WebXOSWallet()
    manager = VialManager(wallet)
    manager.train_vials("test_network", "sample content", "test.txt")
    manager.reset_vials()
    assert manager.get_vials() == {k: {} for k in ["vial1", "vial2", "vial3", "vial4"]}
    assert not manager.network_state