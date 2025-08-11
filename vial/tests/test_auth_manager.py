import pytest
from auth_manager import AuthManager

def test_authenticate():
    auth = AuthManager()
    token, address = auth.authenticate("test_network", "test_session")
    assert isinstance(token, str)
    assert address.startswith("0x")
    assert len(address) == 42
    assert auth.validate_token(token)

def test_validate_session():
    auth = AuthManager()
    token, _ = auth.authenticate("test_network", "test_session")
    assert auth.validate_session(token, "test_network")
    assert not auth.validate_session(token, "wrong_network")

def test_void_session():
    auth = AuthManager()
    token, _ = auth.authenticate("test_network", "test_session")
    auth.void_session(token)
    assert not auth.validate_token(token)