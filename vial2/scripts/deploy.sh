#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r vial2/requirements.txt

echo "Running migrations..."
bash vial2/scripts/migrate.sh

echo "Building Docker image..."
docker build -t vial2-mcp ./vial2

echo "Running container..."
docker run -d -p 8000:8000 --env-file vial2/.env vial2-mcp

echo "Deploying to Netlify..."
netlify deploy --prod --dir . --functions vial2

# xAI Artifact Tags: #vial2 #scripts #deploy #neon_mcp
