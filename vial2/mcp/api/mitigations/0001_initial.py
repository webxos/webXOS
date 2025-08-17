from config.config import DatabaseConfig
import logging

logger = logging.getLogger(__name__)

class Migration:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.project_id = "twilight-art-21036984"

    async def up(self):
        try:
            await self.db.query("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    github_id TEXT,
                    email TEXT,
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    access_token TEXT,
                    expires_at TIMESTAMP,
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS wallets (
                    wallet_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    address TEXT,
                    balance DOUBLE PRECISION,
                    hash TEXT,
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS vials (
                    vial_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    status TEXT,
                    code TEXT,
                    tasks JSONB,
                    config JSONB,
                    wallet_id TEXT REFERENCES wallets(wallet_id),
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    type TEXT,
                    data JSONB,
                    hash TEXT,
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS vial_states (
                    vial_id TEXT PRIMARY KEY,
                    state JSONB,
                    user_id TEXT REFERENCES users(user_id),
                    project_id TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS wallet_transactions (
                    transaction_id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    transaction_type TEXT,
                    amount DECIMAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    project_id TEXT
                );
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                ALTER TABLE users ENABLE ROW LEVEL SECURITY;
                CREATE POLICY users_rls ON users FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
                CREATE POLICY sessions_rls ON sessions FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE wallets ENABLE ROW LEVEL SECURITY;
                CREATE POLICY wallets_rls ON wallets FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE vials ENABLE ROW LEVEL SECURITY;
                CREATE POLICY vials_rls ON vials FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;
                CREATE POLICY blocks_rls ON blocks FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
                CREATE POLICY api_keys_rls ON api_keys FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE vial_states ENABLE ROW LEVEL SECURITY;
                CREATE POLICY vial_states_rls ON vial_states FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                ALTER TABLE wallet_transactions ENABLE ROW LEVEL SECURITY;
                CREATE POLICY wallet_transactions_rls ON wallet_transactions FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
                INSERT INTO migrations (name) VALUES ('0001_initial');
            """)
            logger.info("Migration 0001_initial applied [0001_initial.py:100] [ID:migration_success]")
        except Exception as e:
            logger.error(f"Migration failed: {str(e)} [0001_initial.py:105] [ID:migration_error]")
            raise

    async def down(self):
        try:
            await self.db.query("""
                DROP TABLE IF EXISTS migrations;
                DROP TABLE IF EXISTS wallet_transactions;
                DROP TABLE IF EXISTS vial_states;
                DROP TABLE IF EXISTS api_keys;
                DROP TABLE IF EXISTS blocks;
                DROP TABLE IF EXISTS vials;
                DROP TABLE IF EXISTS wallets;
                DROP TABLE IF EXISTS sessions;
                DROP TABLE IF EXISTS users;
            """)
            logger.info("Migration 0001_initial rolled back [0001_initial.py:115] [ID:rollback_success]")
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)} [0001_initial.py:120] [ID:rollback_error]")
            raise
