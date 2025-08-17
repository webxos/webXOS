import hashlib
import time
from main.api.utils.logging import logger

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            "index": 0,
            "previous_hash": "0",
            "timestamp": time.time(),
            "data": "Genesis Block",
            "nonce": 0
        }
        genesis_block["hash"] = self.calculate_hash(genesis_block)
        self.chain.append(genesis_block)

    def calculate_hash(self, block):
        block_string = f"{block['index']}{block['previous_hash']}{block['timestamp']}{block['data']}{block['nonce']}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def add_block(self, data):
        previous_block = self.chain[-1]
        new_block = {
            "index": len(self.chain),
            "previous_hash": previous_block["hash"],
            "timestamp": time.time(),
            "data": data,
            "nonce": 0
        }
        new_block["hash"] = self.calculate_hash(new_block)
        self.chain.append(new_block)
        logger.info(f"New block added: {new_block['hash']}")
        return new_block["hash"]
