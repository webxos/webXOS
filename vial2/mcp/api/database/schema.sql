-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    github_id TEXT,
    email TEXT,
    project_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enable RLS on users
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY users_rls ON users FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    access_token TEXT,
    expires_at TIMESTAMP,
    project_id TEXT
);

-- Enable RLS on sessions
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY sessions_rls ON sessions FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Wallets table
CREATE TABLE IF NOT EXISTS wallets (
    wallet_id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    address TEXT,
    balance DOUBLE PRECISION,
    hash TEXT,
    project_id TEXT
);

-- Enable RLS on wallets
ALTER TABLE wallets ENABLE ROW LEVEL SECURITY;
CREATE POLICY wallets_rls ON wallets FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Vials table
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

-- Enable RLS on vials
ALTER TABLE vials ENABLE ROW LEVEL SECURITY;
CREATE POLICY vials_rls ON vials FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Blocks table
CREATE TABLE IF NOT EXISTS blocks (
    block_id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    type TEXT,
    data JSONB,
    hash TEXT,
    project_id TEXT
);

-- Enable RLS on blocks
ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;
CREATE POLICY blocks_rls ON blocks FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    api_key TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    project_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enable RLS on api_keys
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY api_keys_rls ON api_keys FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Vial States table
CREATE TABLE IF NOT EXISTS vial_states (
    vial_id TEXT PRIMARY KEY,
    state JSONB,
    user_id TEXT REFERENCES users(user_id),
    project_id TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enable RLS on vial_states
ALTER TABLE vial_states ENABLE ROW LEVEL SECURITY;
CREATE POLICY vial_states_rls ON vial_states FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');

-- Wallet Transactions table
CREATE TABLE IF NOT EXISTS wallet_transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    transaction_type TEXT,
    amount DECIMAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    project_id TEXT
);

-- Enable RLS on wallet_transactions
ALTER TABLE wallet_transactions ENABLE ROW LEVEL SECURITY;
CREATE POLICY wallet_transactions_rls ON wallet_transactions FOR ALL USING (auth.user_id() = user_id AND project_id = 'twilight-art-21036984');
