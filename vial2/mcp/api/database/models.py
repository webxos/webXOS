from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    wallet_address = Column(String(42), unique=True, nullable=False)
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

# xAI Artifact Tags: #vial2 #database #models #neon_mcp
