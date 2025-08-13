import pytest
from vial.webxos_wallet import WebxosWallet
from unittest.mock import patch
import psycopg2

@pytest.fixture
def wallet():
    return WebxosWallet()

def test_create_wallet(wallet):
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_connect.return_value.commit.return_value = None
        wallet_id = wallet.create_wallet("user123")
        assert isinstance(wallet_id, str)
        mock_cursor.execute.assert_called_once()

def test_update_balance(wallet):
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = (100.0,)
        mock_connect.return_value.commit.return_value = None
        balance = wallet.update_balance("wallet123", 50.0)
        assert balance == 150.0
        mock_cursor.execute.assert_called()

def test_update_balance_insufficient(wallet):
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = (10.0,)
        with pytest.raises(ValueError) as exc:
            wallet.update_balance("wallet123", -20.0)
        assert "Insufficient balance" in str(exc.value)

def test_wallet_logging(wallet, tmp_path):
    error_log = tmp_path / "errorlog.md"
    with open(error_log, "a") as f:
        f.write("")
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_connect.return_value.commit.return_value = None
        wallet.create_wallet("user123")
        with open(error_log) as f:
            log_content = f.read()
        assert "Wallet created for user123" in log_content
