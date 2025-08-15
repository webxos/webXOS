-- main/server/mcp/db/schema.sql
-- Schema for hybrid SQL storage to complement MongoDB, ensuring data consistency
-- Tables for users, agents, notes, and wallets with relationships

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id VARCHAR(64) PRIMARY KEY,
    vial_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    status ENUM('stopped', 'running', 'training') DEFAULT 'stopped',
    wallet_address VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notes (
    note_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS note_tags (
    note_id VARCHAR(64) NOT NULL,
    tag VARCHAR(64) NOT NULL,
    PRIMARY KEY (note_id, tag),
    FOREIGN KEY (note_id) REFERENCES notes(note_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wallets (
    wallet_address VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    balance DECIMAL(18,8) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS secrets (
    secret_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    secret_name VARCHAR(255) NOT NULL,
    secret_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_agents_user_id ON agents(user_id);
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_note_tags_tag ON note_tags(tag);
CREATE INDEX idx_wallets_user_id ON wallets(user_id);
CREATE INDEX idx_secrets_user_id ON secrets(user_id);

-- Initial data for testing
INSERT INTO users (user_id, username) VALUES ('test_user_1', 'testuser') ON DUPLICATE KEY UPDATE username = username;
