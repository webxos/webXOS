from typing import Dict
import uuid
import sqlite3
from web3 import Web3
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class WebXOSWallet:
    def __init__(self):
        self.conn = sqlite3.connect("wallet.db")
        self.conn.execute("CREATE TABLE IF NOT EXISTS balances (network_id TEXT PRIMARY KEY, balance REAL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS transactions (id TEXT, network_id TEXT, amount REAL, timestamp TEXT)")
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER")))

    def update_balance(self, network_id: str, amount: float) -> float:
        """Update wallet balance and log transaction."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO balances (network_id, balance) VALUES (?, COALESCE((SELECT balance FROM balances WHERE network_id = ?), 0) + ?)", (network_id, network_id, amount))
            tx_id = str(uuid.uuid4())
            cursor.execute("INSERT INTO transactions (id, network_id, amount, timestamp) VALUES (?, ?, ?, ?)", (tx_id, network_id, amount, "2025-08-11T05:46:00Z"))
            self.conn.commit()
            # Placeholder for Web3 transaction (replace with actual contract address)
            if self.w3.is_connected():
                logger.info(f"Web3 connected, simulating transaction for {amount} $WEBXOS")
            balance = cursor.execute("SELECT balance FROM balances WHERE network_id = ?", (network_id,)).fetchone()[0]
            logger.info(f"Updated balance for {network_id}: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Wallet update error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T05:46:00Z]** Wallet update error: {str(e)}\n")
            raise

    def get_balance(self, network_id: str) -> float:
        """Get balance for a network ID."""
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT balance FROM balances WHERE network_id = ?", (network_id,)).fetchone()
        return result[0] if result else 0.0

    def export_wallet(self, network_id: str) -> str:
        """Export wallet data as markdown."""
        try:
            balance = self.get_balance(network_id)
            cursor = self.conn.cursor()
            transactions = cursor.execute("SELECT id, amount, timestamp FROM transactions WHERE network_id = ?", (network_id,)).fetchall()
            markdown = f"# WEBXOS Wallet\n\nNetwork ID: {network_id}\nBalance: {balance} $WEBXOS\n\n## Transactions\n"
            for tx_id, amount, timestamp in transactions:
                markdown += f"- {timestamp}: {amount} $WEBXOS (ID: {tx_id})\n"
            return markdown
        except Exception as e:
            logger.error(f"Wallet export error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T05:46:00Z]** Wallet export error: {str(e)}\n")
            raise