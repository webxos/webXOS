#!/bin/bash
set -e
mkdir -p uploads/templates uploads/outputs src/agents
cp .env.example .env
echo "OAUTH_CLIENT_SECRET=your_client_secret" >> .env
npm install express ws sqlite3 jsonwebtoken google-auth-library
pip install torch
wget https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py -O src/agents/nanoGPT.py
chmod -R 755 uploads/
touch database.db errorlog.md
chmod 644 database.db errorlog.md
