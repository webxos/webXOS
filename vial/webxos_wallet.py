import sqlite3
import logging
import time
import uuid
import hashlib
import datetime
from web3 import Web3
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebXOSWallet:
    def __init__(self):
        for _ in range(3):
            try:
                self.conn = sqlite3.connect("vial/database.sqlite", timeout=10)
                self.conn.execute("CREATE TABLE IF NOT EXISTS balances (network_id TEXT PRIMARY KEY, balance REAL)")
                self.conn.execute("CREATE TABLE IF NOT EXISTS transactions (id TEXT, network_id TEXT, amount REAL, timestamp TEXT)")
                self.w3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER", "http://localhost:8545")))
                break
            except sqlite3.OperationalError as e:
                logger.error(f"SQLite connection attempt failed: {str(e)}")
                time.sleep(1)
        else:
            raise Exception("Failed to connect to SQLite database after retries")

    def update_balance(self, network_id: str, amount: float) -> None:
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO balances (network_id, balance) VALUES (?, ?)", (network_id, amount))
            self.conn.commit()
            logger.info(f"Updated balance for {network_id}: {amount} $WEBXOS")
        except Exception as e:
            logger.error(f"Balance update error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Balance update error: {str(e)}\n")
            raise

    def get_balance(self, network_id: str) -> float:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT balance FROM balances WHERE network_id = ?", (network_id,))
            result = cursor.fetchone()
            return result[0] if result else 0.0
        except Exception as e:
            logger.error(f"Balance retrieval error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Balance retrieval error: {str(e)}\n")
            raise

    def cashout(self, network_id: str, target_address: str, amount: float) -> bool:
        try:
            balance = self.get_balance(network_id)
            if amount > balance:
                raise ValueError("Insufficient funds")
            cursor = self.conn.cursor()
            cursor.execute("UPDATE balances SET balance = balance - ? WHERE network_id = ?", (amount, network_id))
            tx_id = str(uuid.uuid4())
            cursor.execute("INSERT INTO transactions (id, network_id, amount, timestamp) VALUES (?, ?, ?, ?)", 
                          (tx_id, network_id, -amount, datetime.datetime.utcnow().isoformat()))
            self.conn.commit()
            logger.info(f"Cashed out {amount} $WEBXOS to {target_address} for {network_id}")
            return True
        except Exception as e:
            logger.error(f"Cashout error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Cashout error: {str(e)}\n")
            raise

    def validate_export(self, export_data: str) -> bool:
        try:
            lines = export_data.split('\n')
            if not lines[0].startswith('# WebXOS Vial and Wallet Export'):
                return False
            wallet_section = [line for line in lines if line.startswith('## Wallet')]
            if not wallet_section:
                return False
            vial_sections = [line for line in lines if line.startswith('# Vial Agent: vial')]
            if len(vial_sections) != 4:
                return False
            hash_pattern = r'^[0-9a-f]{64}$'
            for section in vial_sections:
                section_lines = lines[lines.index(section):]
                wallet_hash = next((line.split('Wallet Hash: ')[1] for line in section_lines if line.startswith('Wallet Hash: ')), None)
                if not wallet_hash or not re.match(hash_pattern, wallet_hash):
                    return False
            wallet_hash = next((line.split('Hash: ')[1] for line in lines if line.startswith('Hash: ')), None)
            if not wallet_hash or not re.match(hash_pattern, wallet_hash):
                return False
            return True
        except Exception as e:
            logger.error(f"Export validation error: {str(e)}")
            return False
