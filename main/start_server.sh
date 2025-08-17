#!/bin/bash

set -e

echo "🚀 Starting WEBXOS MCP Gateway..."

# Check if .env exists
if [ ! -f "main/.env" ]; then
    echo "⚠️  Creating .env from example..."
    cp main/.env.example main/.env
fi

# Install dependencies
echo "📦 Installing dependencies..."
cd main
pip install -r requirements.txt

# Start services
echo "🔧 Starting Redis (if needed)..."
redis-server --daemonize yes --port 6379 || echo "Redis may already be running"

echo "🔧 Starting MongoDB (if needed)..."
mongod --fork --logpath /tmp/mongod.log --dbpath /tmp/mongodb || echo "MongoDB may already be running"

# Run FastAPI server
echo "🌐 Starting FastAPI server..."
export PYTHONPATH=$(pwd)
uvicorn main.api.main:app --host 0.0.0.0 --port 8000 --reload --workers 1

echo "✅ Server started at http://localhost:8000"
echo "📊 Health check: curl http://localhost:8000/v1/health"
echo "🖥️  Frontend: http://localhost:8000 (if serving static files)"
