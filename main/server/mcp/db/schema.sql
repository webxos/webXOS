-- Schema for PostgreSQL and MySQL databases for Vial MCP

-- Users table for storing user information
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    wallet_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSON
);

-- Wallets table for storing wallet details
CREATE TABLE IF NOT EXISTS wallets (
    wallet_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    api_key VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Notes table for storing user notes
CREATE TABLE IF NOT EXISTS notes (
    note_id VARCHAR(255) PRIMARY KEY,
    wallet_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    resource_id VARCHAR(255),
    db_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

-- Resources table for storing resource metadata
CREATE TABLE IF NOT EXISTS resources (
    resource_id VARCHAR(255) PRIMARY KEY,
    wallet_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    db_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

-- Quantum links table for storing quantum link data
CREATE TABLE IF NOT EXISTS quantum_links (
    link_id VARCHAR(255) PRIMARY KEY,
    wallet_id VARCHAR(255) NOT NULL,
    vial_id VARCHAR(255) NOT NULL,
    db_type VARCHAR(50) NOT NULL,
    content JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

-- Audit logs table for tracking actions
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    wallet_id VARCHAR(255),
    endpoint VARCHAR(255) NOT NULL,
    action VARCHAR(255) NOT NULL,
    details JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

-- Performance metrics table for tracking API performance
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    wallet_id VARCHAR(255),
    response_time FLOAT NOT NULL,
    status_code INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);
