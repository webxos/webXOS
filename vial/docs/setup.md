Vial MCP Controller Setup
Prerequisites

Node.js: v16 or higher
Python: 3.9 or higher
Docker: For containerized deployment
Vercel CLI: For edge deployment
GitHub Repository: github.com/webxos/webxos

Installation

Clone Repository:git clone https://github.com/webxos/webxos.git
cd webxos


Install Dependencies:npm install
pip install -r db/requirements.txt


Set Environment Variables:
Copy .env.example to .env and update with API keys:NOMIC_API_KEY=your_nomic_key
COGNITAL_API_KEY=your_cognital_key
LLMWARE_API_KEY=your_llmware_key
JINAAI_API_KEY=your_jinaai_key
JWT_SECRET=VIAL_MCP_SECRET_2025
OAUTH_CLIENT_ID=your_github_client_id
OAUTH_CLIENT_SECRET=your_github_client_secret





Deployment

Local Deployment:docker-compose up -d


Access frontend at http://localhost:3000.
Access Streamlit dashboard at http://localhost:8501.


Vercel Deployment:vercel --prod


Configure vercel.json for routes and environment variables.
Ensure SSL/TLS is enabled (handled by Vercel).


GitHub Actions:
CI/CD pipeline in ci.yml automates testing and deployment.
Push to main branch triggers Vercel deployment.



Configuration

Database:
PostgreSQL: Run db/mcp_db_init.py to initialize schema.
SQLite: Legacy support via vial/database.sqlite.
MongoDB, Milvus, Weaviate, FAISS: Configured in db/library_config.json.


Agents:
Nomic, CogniTALLMware, LLMware, Jina AI configured in resources/mcp_tools.json.


Frontend:
Use vial.js, chatbot.js, and static assets (style.css, neurots.js, icon.png, redaxios.min.js).


Backend:
Run vial/unified_server.py for API Gateway.
Node.js server (node/server.js) for OAuth and network sync.



Testing

Run tests:pytest vial/tests
npm test


Verify endpoints (/v1/api/retrieve, /v1/api/llm, /v1/api/git) return 200 OK.

Troubleshooting

Check db/errorlog.md for database and API errors.
Check vial/errorlog.md for vial-specific errors.
Ensure nginx.conf is configured for SSL/TLS.
Verify API keys and OAuth credentials in .env.
