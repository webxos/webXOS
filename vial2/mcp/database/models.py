from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from ..database.neon_config import neon_config
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class VialLog(Base):
    __tablename__ = "vial_logs"
    log_id = Column(Integer, primary_key=True)
    vial_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    event_data = Column(JSON)
    node_id = Column(String)
    created_at = Column(DateTime, nullable=False)

engine = create_async_engine(neon_config.config["database_url"], echo=True)

async def init_models():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database models initialized")
    except Exception as e:
        logger.error(f"Database models initialization failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #database #models #neon #neon_mcp
