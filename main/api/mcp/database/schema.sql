CREATE TABLE IF NOT EXISTS users (
  user_id UUID PRIMARY KEY,
  api_key VARCHAR(255) NOT NULL UNIQUE,
  api_secret VARCHAR(255) NOT NULL,
  balance DECIMAL(18,4) DEFAULT 0.0000,
  reputation BIGINT DEFAULT 0,
  wallet_address VARCHAR(255) NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vials (
  id VARCHAR(50) PRIMARY KEY,
  user_id UUID REFERENCES users(user_id),
  status VARCHAR(50) DEFAULT 'Stopped',
  balance DECIMAL(18,4) DEFAULT 0.0000,
  wallet_address VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
  transaction_id VARCHAR(255) PRIMARY KEY,
  from_address VARCHAR(255) NOT NULL,
  to_address VARCHAR(255) NOT NULL,
  amount DECIMAL(18,4) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS blockchain (
  block_id VARCHAR(255) PRIMARY KEY,
  previous_hash VARCHAR(255) NOT NULL,
  hash VARCHAR(255) NOT NULL,
  data JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_id ON vials(user_id);
CREATE INDEX idx_transaction_addresses ON transactions(from_address, to_address);
CREATE INDEX idx_block_hash ON blockchain(hash);
