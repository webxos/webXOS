WebXOS Searchbot Backend
This directory (/chatbot/db/) contains the backend infrastructure for the WebXOS Searchbot, a lightweight, standalone AI-powered search and interaction platform. The backend uses FastAPI, MongoDB 8.0, and zero-dependency native libraries to handle JSON requests, logging, and automated storage for the chatbot, vial modules, and WebXOS ecosystem.
Directory Contents

server.py: FastAPI server handling JSON endpoints for authentication, vials, queries, wallet, imports, and modules.
nlp_model.py: Query enhancement logic for natural language processing.
mcp_db_init.py: Initializes the MCP database (mcp_db) with collections: users, queries, errors, wallet, vials, modules.
Dockerfile: Builds the backend container with Python 3.9 and MongoDB dependencies.
requirements.txt: Lists Python dependencies (fastapi, uvicorn, pymongo, aiofiles, requests, beautifulsoup4).
wait-for-it.sh: Ensures MongoDB is available before starting the server.
README.md: This file.

Setup

Prerequisites:

Docker: Required for running MongoDB and the backend.
MongoDB 8.0: Run via Docker (mongodb/mongodb-community-server:8.0-ubi8).
Python 3.9+: For local development (optional).


Start MongoDB:
docker run -d --name mongo -p 27017:27017 mongodb/mongodb-community-server:8.0-ubi8


Initialize MCP Database:
python mcp_db_init.py


Build and Run Backend:
docker build -t searchbot-backend .
docker run -d --name searchbot-backend --link mongo:mongo -p 8000:8000 searchbot-backend


Verify:
curl http://localhost:8000/api/health
# {"status":"healthy"}
curl -X POST -H "Content-Type: application/json" -d '{"userId":"testuser"}' http://localhost:8000/api/auth
# {"apiKey":"..."}



Endpoints

GET /api/health: Check MongoDB connectivity.
POST /api/auth: Generate API key for a user.
GET /api/vials: Retrieve vial configurations (requires auth).
POST /api/log_query: Log queries to mcp_db.queries (requires auth).
POST /api/wallet: Update wallet in mcp_db.wallet (requires auth).
POST /api/enhance_query: Enhance queries using nlp_model.py (requires auth).
POST /api/import: Import nano_gpt_bots.md to mcp_db.vials (requires auth).
POST /api/modules: Add WebXOS modules to mcp_db.modules (requires auth).

MCP Database
The MCP database (mcp_db) centralizes data:

users: Stores user IDs and API keys.
queries: Logs queries with timestamps and user IDs.
errors: Logs errors from frontend and backend.
wallet: Tracks webxos balance and transactions.
vials: Stores vial configurations (from nano_gpt_bots.md).
modules: Stores future WebXOS module data.

Notes

Offline Support: The frontend (chatbot.html) supports simulation mode if MongoDB or the backend is unavailable.
Logging: Errors and queries are stored in mcp_db.errors and mcp_db.queries.
Extensibility: Add new modules via /api/modules for WebXOS integration.
Dependencies: Minimal, using native Python libraries and MongoDB Node.js driver in the frontend.

For frontend setup, see /chatbot/chatbot.html and /chatbot/nano_gpt_bots.md for vial configurations.
