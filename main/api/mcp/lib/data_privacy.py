from fastapi import APIRouter, Depends, HTTPException
from config.config import DatabaseConfig
from lib.security import SecurityHandler
from pydantic import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger("mcp.data_privacy")
logger.setLevel(logging.INFO)

router = APIRouter()

class DataErasureInput(BaseModel):
    user_id: str

class DataErasureOutput(BaseModel):
    status: str
    deleted_tables: list

class DataPrivacyHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security_handler = SecurityHandler(db)

    async def erase_user_data(self, input: DataErasureInput) -> DataErasureOutput:
        try:
            user = await self.db.query(
                "SELECT user_id FROM users WHERE user_id = $1",
                [input.user_id]
            )
            if not user.rows:
                raise HTTPException(status_code=404, detail="User not found")

            deleted_tables = []
            
            # Delete from sessions
            await self.db.query(
                "DELETE FROM sessions WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("sessions")
            
            # Delete from security_events
            await self.db.query(
                "DELETE FROM security_events WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("security_events")
            
            # Delete from transactions
            await self.db.query(
                "DELETE FROM transactions WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("transactions")
            
            # Delete from vials
            await self.db.query(
                "DELETE FROM vials WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("vials")
            
            # Delete from quantum_links
            await self.db.query(
                "DELETE FROM quantum_links WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("quantum_links")
            
            # Delete from users
            await self.db.query(
                "DELETE FROM users WHERE user_id = $1",
                [input.user_id]
            )
            deleted_tables.append("users")
            
            await self.security_handler.log_event(
                event_type="data_erasure",
                user_id=input.user_id,
                details={"deleted_tables": deleted_tables}
            )
            logger.info(f"Erased data for user {input.user_id}")
            return DataErasureOutput(status="success", deleted_tables=deleted_tables)
        except Exception as e:
            logger.error(f"Error erasing user data: {str(e)}")
            await self.security_handler.log_event(
                event_type="data_erasure_error",
                user_id=input.user_id,
                details={"error": str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/privacy/erase")
async def erase_data(input: DataErasureInput, handler: DataPrivacyHandler = Depends(lambda: DataPrivacyHandler(DatabaseConfig()))):
    return await handler.erase_user_data(input)
