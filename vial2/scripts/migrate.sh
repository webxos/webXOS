#!/bin/bash
set -e

echo "Applying database migrations..."
psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, wallet_address VARCHAR(42) UNIQUE NOT NULL, api_key VARCHAR(256), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS wallets (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), address VARCHAR(42) UNIQUE NOT NULL, balance FLOAT DEFAULT 0.0, hash VARCHAR(64), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS vials (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), vial_id VARCHAR(10) NOT NULL, status VARCHAR(20) DEFAULT 'stopped', code TEXT, code_length INTEGER DEFAULT 0, webxos_hash VARCHAR(64), wallet_id INTEGER REFERENCES wallets(id), config JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS computes (id SERIAL PRIMARY KEY, compute_id VARCHAR(20) NOT NULL, state VARCHAR(20) DEFAULT 'Empty', spec JSON, readiness BOOLEAN DEFAULT FALSE, last_activity TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
psql $DATABASE_URL -c "CREATE TABLE IF NOT EXISTS logs (id SERIAL PRIMARY KEY, event_type VARCHAR(50), message TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"

echo "Setting up RLS policies..."
psql $DATABASE_URL -c "ALTER TABLE users ENABLE ROW LEVEL SECURITY;"
psql $DATABASE_URL -c "CREATE POLICY user_access ON users FOR ALL TO PUBLIC USING (true);"
psql $DATABASE_URL -c "ALTER TABLE wallets ENABLE ROW LEVEL SECURITY;"
psql $DATABASE_URL -c "CREATE POLICY wallet_access ON wallets FOR ALL TO PUBLIC USING (true);"

echo "Database migration complete."

# xAI Artifact Tags: #vial2 #scripts #migrate #neon_mcp
