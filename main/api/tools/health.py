from config.config import DatabaseConfig
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any

logger = logging.getLogger("mcp.health")
logger.setLevel(logging.INFO)

class HealthOutput(BaseModel):
    status: str
    database: str
    tools_available: list

class HealthTool:
    def __init__(self, db: DatabaseConfig, tools: Dict[str, Any]):
        self.db = db
        self.tools = tools

    async def execute(self, input: Dict[str, Any]) -> HealthOutput:
        try:
            # Check database connectivity
            await self.db.query("SELECT 1")
            database_status = "connected"
            
            # List available tools
            tools_list = list(self.tools.keys())
            
            logger.info("Health check successful")
            return HealthOutput(
                status="ok",
                database=database_status,
                tools_available=tools_list
            )
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            raise HTTPException(500, f"Health check failed: {str(e)}")
