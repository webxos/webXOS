CREATE TABLE users (
    user_id VARCHAR(255) PRIMARY KEY,
    balance FLOAT NOT NULL DEFAULT 0.0,
    wallet_address VARCHAR(255) NOT NULL,
    api_key VARCHAR(255),
    api_secret VARCHAR(255),
    reputation INTEGER NOT NULL DEFAULT 0,
    access_token VARCHAR(255)  -- Added for OAuth token storage
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
