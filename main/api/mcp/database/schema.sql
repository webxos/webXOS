CREATE TABLE users (
    user_id VARCHAR(255) PRIMARY KEY,
    balance FLOAT NOT NULL DEFAULT 0.0,
    wallet_address VARCHAR(255) NOT NULL,
    api_key VARCHAR(255),
    api_secret VARCHAR(255),
    reputation INTEGER NOT NULL DEFAULT 0,
    access_token VARCHAR(255)
);

CREATE TABLE vials (
    vial_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    code TEXT NOT NULL,
    wallet_address VARCHAR(255) NOT NULL,
    webxos_hash VARCHAR(255) NOT NULL,
    PRIMARY KEY (vial_id, user_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE quantum_links (
    link_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    quantum_state JSONB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE transactions (
    transaction_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    amount FLOAT NOT NULL,
    destination_address VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_key VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id),
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    client_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
