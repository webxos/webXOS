CREATE EXTENSION IF NOT EXISTS neon;

CREATE TABLE IF NOT EXISTS agents (
    vial_id VARCHAR(50) PRIMARY KEY,
    training_data JSONB[],
    model_weights JSONB,
    model_type VARCHAR(50),
    timestamp TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vial_id ON agents (vial_id);
