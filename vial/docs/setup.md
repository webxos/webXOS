Setup Instructions
Overview
This guide provides step-by-step instructions to set up the Vial MCP Controller for local development and deployment.
Prerequisites

Python 3.9+
Node.js 16+
Docker and Docker Compose
MongoDB and Redis (or use Docker services)
Git

Installation

Clone the Repository:git clone https://github.com/webxos/webxos.git
cd webxos


Install Python Dependencies:pip install -r db/requirements.txt


Install Node.js Dependencies:npm install --prefix node


Configure Environment:
Copy .env.example to .env and set variables:MONGO_URI=mongodb://localhost:27017
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET=VIAL_MCP_SECRET_2025
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
WALLET_INCREMENT=0.0001




Initialize Database:python db/mcp_db_init.py


Start Services:docker-compose up -d


Run Tests:pytest db/ vial/tests/



Accessing the Application

Main UI: http://localhost:8000/vial.html
Chatbot UI: http://localhost:8000/chatbot.html
Secondary Chatbot UI: http://localhost:8000/chatbot/chatbot2.html
API: http://localhost:8000/docs (OpenAPI interface)

Deployment

Netlify: Use netlify.toml for deployment to webxos.netlify.app.
GitHub Actions: Configure ci.yml for CI/CD.
Docker: Build and run using Dockerfile and docker-compose.yaml.

Troubleshooting

Check db/errorlog.md for database and backend errors.
Check errorlog.md for conversation-related errors.
Ensure MongoDB and Redis are running (docker ps).

For API details, see /docs/api.markdown.
