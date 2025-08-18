CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    address VARCHAR(42) UNIQUE NOT NULL,
    balance DECIMAL(18, 4) DEFAULT 0.0,
    hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vials (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    vial_id VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'stopped',
    code TEXT,
    code_length INTEGER DEFAULT 0,
    webxos_hash VARCHAR(64),
    wallet_id INTEGER REFERENCES wallets(id),
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS computes (
    id SERIAL PRIMARY KEY,
    compute_id VARCHAR(20) NOT NULL,
    state VARCHAR(20) DEFAULT 'Empty',
    spec JSONB,
    readiness BOOLEAN DEFAULT FALSE,
    last_activity TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    event
