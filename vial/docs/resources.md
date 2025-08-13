Vial MCP Controller Setup Instructions
Prerequisites

Docker and Docker Compose: For containerized deployment.
Python 3.8+: For backend scripts and testing.
Node.js 16+: For authentication server and frontend tests.
MongoDB: For backend data storage.
Redis 7.0: For session caching.
Netlify CLI: For deployment.

Installation

Clone the Repository:
git clone <repository_url>
cd vial-mcp-project


Set Up Environment Variables:
cp .env.example .env

Edit .env to include:

WEB3_PROVIDER: Web3 provider URL (e.g., http://localhost:8545).
JWT_SECRET: Secret for JWT authentication.
MONGO_URL: MongoDB connection string (e.g., mongodb://mongo:27017).
REDIS_URL: Redis connection string (e.g., redis://redis:6379).


Build and Run with Docker:
docker-compose up --build


Run Locally:

Start FastAPI server:uvicorn vial.unified_server:app --host 0.0.0.0 --port 8000


Start Node.js server:cd node && npm start


Access vial.html at http://localhost:8000/vial.html.



Deployment

Deploy to Netlify:netlify deploy --prod


Configure netlify.toml for static assets and serverless functions.

Testing
Run unit and integration tests:
pytest vial/tests/
node vial/tests/test_vial_html.js

Notes

Ensure the chatbot folder remains unchanged.
Wallet exports (vial_wallet_export_*.md) must follow the specified format with four vials.
Logs are written to vial/errorlog.md for debugging.
