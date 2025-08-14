import logging,sqlite3,os
from datetime import datetime

logger=logging.getLogger(__name__)

class MCPDatabaseInitializer:
    """Initializes SQLite database for Vial MCP."""
    def __init__(self,db_path="/app/vial_mcp.db"):
        """Initialize MCPDatabaseInitializer with database path.

        Args:
            db_path (str): Path to SQLite database.
        """
        self.db_path=db_path
        logger.info("MCPDatabaseInitializer initialized")

    def initialize_database(self):
        """Create necessary SQLite tables for authentication, notes, quantum states, and wallets."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                # Wallets table
                cursor.execute("""CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    timestamp TEXT NOT NULL)""")
                # Notes table
                cursor.execute("""CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    resource_id TEXT,
                    timestamp TEXT NOT NULL,
                    wallet_id TEXT NOT NULL)""")
                # Quantum states table
                cursor.execute("""CREATE TABLE IF NOT EXISTS quantum_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vial_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    wallet_id TEXT NOT NULL)""")
                conn.commit()
                logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            with open("/app/errorlog.md","a") as f:
                f.write(f"[{datetime.now().isoformat()}] [MCPDatabaseInitializer] Database initialization failed: {str(e)}\n")
            raise Exception(f"Database initialization failed: {str(e)}")

    def seed_initial_data(self):
        """Seed initial data for testing purposes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO wallets (wallet_id,user_id,api_key,timestamp) VALUES (?,?,?,?)",
                              ("wallet_123","user_123","api-a24cb96b-96cd-488d-a013-91cb8edbbe68",datetime.now().isoformat()))
                conn.commit()
                logger.info("Initial data seeded successfully")
        except Exception as e:
            logger.error(f"Initial data seeding failed: {str(e)}")
            with open("/app/errorlog.md","a") as f:
                f.write(f"[{datetime.now().isoformat()}] [MCPDatabaseInitializer] Initial data seeding failed: {str(e)}\n")
            raise Exception(f"Initial data seeding failed: {str(e)}")
