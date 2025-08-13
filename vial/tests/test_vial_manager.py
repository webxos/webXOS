import pytest
from vial.vial_manager import VialManager
from pymongo import MongoClient

@pytest.fixture
def vial_manager():
    client = MongoClient('mongodb://mongo:27017')
    db = client['mcp_db']
    db.vials.drop()
    return VialManager()

def test_validate_vials(vial_manager):
    valid_vials = {
        "vial1": {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial2": {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial3": {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"},
        "vial4": {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"}
    }
    assert vial_manager.validate_vials(valid_vials)
    invalid_vials = {
        "vial1": {"status": "running", "script": "code", "wallet_hash": "invalid"},
        "vial2": {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"}
    }
    assert not vial_manager.validate_vials(invalid_vials)

def test_update_vial(vial_manager):
    vial_data = {"status": "running", "script": "code", "wallet_hash": "bfcabdd444109c56f26d65b300e407ee6e26dc122648649206b8ea433caf3e95"}
    assert vial_manager.update_vial("vial1", vial_data)
    result = vial_manager.db.vials.find_one({"id": "vial1"})
    assert result["status"] == "running"
