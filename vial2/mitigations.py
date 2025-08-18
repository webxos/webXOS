from sqlalchemy import create_engine
from .database import Base
from .config import config

def run_migration():
    engine = create_engine(config.DATABASE_URL)
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    run_migration()

# xAI Artifact Tags: #vial2 #migrations #neon_mcp
