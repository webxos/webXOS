-- Initialize sample blockchain data (based on vial_wallet_export_2025-08-17T11-50-01-797Z.md)
INSERT INTO blocks (hash, created_at) VALUES
('abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab', CURRENT_TIMESTAMP - INTERVAL '2 days'),
('def4567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef', CURRENT_TIMESTAMP - INTERVAL '1 day'),
('ghi7890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123', CURRENT_TIMESTAMP);

-- Initialize sample user data
INSERT INTO users (user_id, username, wallet_address, balance, reputation, created_at) VALUES
('user_12345', 'test_user', 'wallet_user_12345', 100.0, 1000, CURRENT_TIMESTAMP),
('user_67890', 'claude_user', 'wallet_user_67890', 50.0, 500, CURRENT_TIMESTAMP);

-- Initialize sample session data
INSERT INTO sessions (user_id, access_token, expires_at, created_at) VALUES
('user_12345', 'sample_jwt_token_12345', CURRENT_TIMESTAMP + INTERVAL '1 day', CURRENT_TIMESTAMP),
('user_67890', 'sample_jwt_token_67890', CURRENT_TIMESTAMP + INTERVAL '1 day', CURRENT_TIMESTAMP);
