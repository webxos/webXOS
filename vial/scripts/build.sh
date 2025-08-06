#!/bin/bash
set -e
npm install express ws sqlite3 jsonwebtoken google-auth-library
pip install torch
wget https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py -O src/agents/nanoGPT.py
chmod -R 755 uploads/
touch database.db errorlog.md
chmod 644 database.db errorlog.md
node src/server.js &
