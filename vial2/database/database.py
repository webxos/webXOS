from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
from fastapi import Depends
import asyncpg
from ..config import config
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    wallet_address = Column(String(42), unique=True, nullable=False)
    api_key = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)

class Wallet(Base):
    __tablename__ = 'wallets'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    address = Column(String(42), unique=True, nullable=False)
    balance = Column(Float, default=0.0)
    hash = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)

class Vial(Base):
    __tablename__ = 'vials'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    vial_id = Column(String(10), nullable=False)
    status = Column(String(20), default='stopped')
    code = Column(String)
    code_length = Column(Integer, default=0)
    webxos_hash = Column(String(64))
    wallet_id = Column(Integer, ForeignKey('wallets.id'))
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Compute(Base):
    __tablename__ = 'computes'
    id = Column(Integer, primary_key=True)
    compute_id = Column(String(20), nullable=False)
    state = Column(String(20), default='Empty')
    spec = Column(JSON)
    readiness = Column(Boolean, default=False)
    last_activity = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50))
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def get_db():
    try:
        conn = await asyncpg.connect(config.DATABASE_URL)
        yield conn
    except Exception as e:
        error_logger.log_error("database", str(e), str(e.__traceback__))
        raise
    finally:
        await conn.close()

async def execute_query(db, query: str, params: dict = None):
    try:
        result = await db.execute(text(query), params or {})
        if query.strip().upper().startswith("SELECT"):
            rows = await db.fetchall()
            return [dict(row) for row in rows]
        return {"status": "success"}
    except Exception as e:
        error_logger.log_error("execute_query", str(e), str(e.__traceback__))
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #database #neon_mcp
