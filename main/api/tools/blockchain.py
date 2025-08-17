from config.config import DatabaseConfig
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
from lib.errors import ValidationError

logger = logging.getLogger("mcp.blockchain")
logger.setLevel(logging.INFO)

class BlockchainInfoOutput(BaseModel):
    block_count: int
    last_hash: str

class BlockchainTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "getBlockchainInfo")
            if method == "getBlockchainInfo":
                return await self.get_blockchain_info()
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Blockchain error: {str(e)}")
            raise HTTPException(400, str(e))

    async def get_blockchain_info(self) -> BlockchainInfoOutput:
        try:
            # Placeholder query; assumes a blocks table exists
            result = await self.db.query("SELECT COUNT(*) as count, MAX(hash) as last_hash FROM blocks")
            if not result.rows:
                raise HTTPException(404, "No blockchain data found")
            data = result.rows[0]
            logger.info("Retrieved blockchain info")
            return BlockchainInfoOutput(
                block_count=data["count"],
                last_hash=data["last_hash"] or "0" * 64
            )
        except Exception as e:
            logger.error(f"Get blockchain info error: {str(e)}")
            raise HTTPException(400, str(e))
