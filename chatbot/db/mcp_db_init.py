import psycopg2
import os
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_postgres():
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/mcp_db"))
        cur = conn.cursor()

        # Create tables for API keys, RBAC policies, and wallet
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                user_id VARCHAR(255) PRIMARY KEY,
                api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rbac_policies (
                user_id VARCHAR(255),
                role VARCHAR(255),
                PRIMARY KEY (user_id, role)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                user_id VARCHAR(255) PRIMARY KEY,
                webxos FLOAT DEFAULT 0.0,
                transactions JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                user_id VARCHAR(255),
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS vectors (
                user_id VARCHAR(255),
                vector VECTOR(768),
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logger.info("PostgreSQL tables initialized successfully")
    except Exception as e:
        logger.error(f"PostgreSQL initialization error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** PostgreSQL initialization error: {str(e)}\n")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    init_postgres()
