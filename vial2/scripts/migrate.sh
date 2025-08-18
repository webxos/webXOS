#!/bin/bash

export PGPASSWORD=$NEON_DB_PASSWORD
psql -h pg.neon.tech -U neondb_owner -d neondb -f vial2/database/schema.sql
python vial2/migrations.py

# xAI Artifact Tags: #vial2 #migrate #neon_mcp
