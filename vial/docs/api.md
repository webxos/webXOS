API Documentation
Overview
The Vial MCP Controller API provides endpoints for managing vials, processing queries, and handling wallet transactions. All endpoints are secured with JWT authentication (vial/auth_manager.py) and integrate with the wallet system (vial/webxos_wallet.py).
Endpoints
Authentication

POST /api/authenticate
Description: Authenticates a user and returns a JWT token.
Request Body: { "user_id": string, "password": string }
Response: { "token": string }
Logs: Stored in auth_logs collection (MongoDB).



Database Queries

POST /api/query
Description: Processes database queries, updating wallet.
Request Body: { "user_id": string, "query": string, "wallet": { "webxos": float, "transactions": array } }
Response: { "status": string, "wallet": object }
Logs: Stored in query_logs collection.



Chatbot Queries

POST /chatbot/api/query
Description: Processes chatbot queries using nano-GPT (chatbot/server.py).
Request Body: { "user_id": string, "query": string, "wallet": object }
Response: { "response": string, "wallet": object }
Logs: Stored in chatbot_logs collection.



Vial Management

POST /api/manage_vial
Description: Manages vial commands for Nomic, CogniTALLMware, LLMware, Jina AI.
Request Body: { "user_id": string, "vial_id": string, "command": string, "wallet": object }
Response: { "status": string, "vial_id": string, "wallet": object }
Logs: Stored in vial_logs collection.



Wallet Updates

POST /api/update_wallet
Description: Updates user wallet with transactions.
Request Body: { "user_id": string, "transaction": object }
Response: { "webxos": float, "transactions": array }
Logs: Stored in wallet collection and SQLite (vial/database.sqlite).



LangChain Queries

POST /api/langchain_query
Description: Processes queries using LangChain (vial/langchain_agent.py).
Request Body: { "user_id": string, "query": string, "wallet": object }
Response: { "response": string, "wallet": object }
Logs: Stored in langchain_logs collection.



Error Handling
Errors are logged to db/errorlog.md with timestamps to prevent redundancy.
Deployment

Deployed via netlify.toml on webxos.netlify.app.
CI/CD configured in ci.yml for GitHub Actions.
Docker setup in Dockerfile and docker-compose.yaml.

For setup instructions, see /docs/setup.markdown.
