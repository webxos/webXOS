CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    wallet_address VARCHAR(255) UNIQUE NOT NULL,
    balance DECIMAL(10,2) DEFAULT 0.0,
    reputation INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    access_token TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS blocks (
    block_id SERIAL PRIMARY KEY,
    hash VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS code_executions (
    execution_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    code TEXT NOT NULL,
    output TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quantum_links (
    link_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
