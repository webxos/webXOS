import uuid
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class WebXOSWallet:
    def __init__(self):
        self.addresses: Dict[str, float] = {}

    def create_address(self) -> str:
        try:
            address = str(uuid.uuid4())
            self.addresses[address] = 0.0
            logger.info(f"Created wallet address: {address}")
            return address
        except Exception as e:
            logger.error(f"Wallet address creation error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-10T20:23:00Z]** Wallet address creation error: {str(e)}\n")
            raise

    def add_balance(self, address: str, amount: float) -> None:
        try:
            if address in self.addresses:
                self.addresses[address] += amount
                logger.info(f"Added {amount} to address: {address}")
            else:
                raise ValueError("Invalid wallet address")
        except Exception as e:
            logger.error(f"Add balance error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-10T20:23:00Z]** Add balance error: {str(e)}\n")
            raise

    def get_balance(self, address: str) -> float:
        try:
            balance = self.addresses.get(address, 0.0)
            logger.info(f"Retrieved balance for address: {address}")
            return balance
        except Exception as e:
            logger.error(f"Get balance error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-10T20:23:00Z]** Get balance error: {str(e)}\n")
            raise
