CREATE TABLE IF NOT EXISTS wallets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    wallet_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS notes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    content TEXT NOT NULL,
    resource_id VARCHAR(255),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    wallet_id VARCHAR(255) NOT NULL,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

CREATE TABLE IF NOT EXISTS quantum_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vial_id VARCHAR(255) NOT NULL,
    state TEXT NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    wallet_id VARCHAR(255) NOT NULL,
    FOREIGN KEY (wallet_id) REFERENCES wallets(wallet_id)
);

INSERT IGNORE INTO wallets (wallet_id, user_id, api_key, timestamp)
VALUES ('wallet_123', 'user_123', 'api-a24cb96b-96cd-488d-a013-91cb8edbbe68', CURRENT_TIMESTAMP);
