import pytest
from vial.auth_manager import AuthManager
import jwt
import os

@pytest.fixture
def auth_manager():
    os.environ["JWT_SECRET"] = "testsecret"
    return AuthManager()

def test_generate_token(auth_manager):
    token = auth_manager.generate_token("test_user")
    assert token
    payload = jwt.decode(token, "testsecret", algorithms=["HS256"])
    assert payload["userId"] == "test_user"

def test_verify_token(auth_manager):
    token = auth_manager.generate_token("test_user")
    payload = auth_manager.verify_token(token)
    assert payload["userId"] == "test_user"

def test_verify_invalid_token(auth_manager):
    with pytest.raises(Exception):
        auth_manager.verify_token("invalid_token")
