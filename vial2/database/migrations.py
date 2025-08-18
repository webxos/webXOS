from sqlalchemy import create_engine
from ..database.database import Base
from ..config import config
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def run_migration():
    try:
        engine = create_engine(config.DATABASE_URL)
        Base.metadata.create_all(engine)
        logger.info("Database migration completed successfully")
    except Exception as e:
        error_logger.log_error("migrations", f"Database migration failed: {str(e)}", str(e.__traceback__))
        logger.error(f"Database migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_migration()

# xAI Artifact Tags: #vial2 #migrations #neon_mcp
