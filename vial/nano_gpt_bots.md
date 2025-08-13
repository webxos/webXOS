Nano GPT Bots Documentation
Overview
The chatbot folder contains frontend and backend components for AI-driven chatbots integrated with the Vial MCP Controller.
Files

chatbot2.html: Secondary chatbot UI for alternative interactions.
server.py: Backend for chatbot operations, handling requests and responses.
site_index.json: Search index for chatbot tools and pages.
sw.js: Service worker for offline support.

Integration

Chatbots interact with vial/unified_server.py via /api/vials endpoint.
Use vial/langchain_agent.py for query enhancement.
Wallet operations are validated using vial/webxos_wallet.py.

Setup

Ensure node/server.js is running for authentication.
Configure .env with JWT_SECRET for secure API access.
Run docker-compose up to start MongoDB and Redis dependencies.

Error Handling

Errors are logged to vial/errorlog.md.
Check MongoDB connectivity in db/mcp_db_init.py.
