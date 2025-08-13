Vial MCP Controller Resources
Databases

PostgreSQL: Stores API keys, RBAC policies, wallet data, and user data. Configured in db/mcp_db_init.py. URI: postgresql://user:password@localhost:5432/mcp_db.
MongoDB: Handles unstructured data for user sessions. URI: mongodb://localhost:27017.
Milvus: Vector database for Nomic and CogniTALLMware embeddings. URI: localhost:19530.
Weaviate: Vector database for LLMware and Jina AI embeddings. URI: http://localhost:8080.
pgvector: PostgreSQL extension for vector storage. Configured in db/mcp_db_init.py.
FAISS: Local vector index for fast similarity search, integrated via db/library_agent.py.

LLMs

LLaMA 3.3: General-purpose LLM, accessed via HuggingFace API.
Mistral: Optimized for text generation, integrated in vial/langchain_agent.py.
Gemma 2: Lightweight LLM for efficient inference.
Qwen: Multilingual LLM for diverse queries.
Phi: Compact LLM for resource-constrained environments.

Agents

Nomic (agent1.py): Vector-based search using Nomic embeddings. Endpoint: /v1/api/nomic_search.
CogniTALLMware (agent2.py): Vector-based search with CogniTALLMware embeddings. Endpoint: /v1/api/cognitallmware_search.
LLMware (agent3.py): Vector-based search with LLMware embeddings. Endpoint: /v1/api/llmware_search.
Jina AI (agent4.py): Vector-based search with Jina AI embeddings. Endpoint: /v1/api/jinaai_search.

Other Resources

Redis: Caching for API responses and metrics. Host: localhost:6379.
Nginx: Reverse proxy with SSL/TLS, configured in nginx.conf.
Vercel: Edge deployment for frontend (vial.js, chatbot.js) and API routes (/v1/api/*).
GitHub: Repository at github.com/webxos/webxos for Git operations.
Streamlit: Dashboard for API metrics and data retrieval at http://localhost:8501.

Deployment Notes

Ensure environment variables are set in .env (see docs/vial_readme.md).
Verify SSL/TLS certificates in nginx.conf for secure deployment.
Run docker-compose up -d to start services.
Deploy to Vercel using ci.yml with GitHub Actions.
