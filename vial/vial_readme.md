Vial MCP Controller
Overview
The Vial MCP Controller is a master control panel for managing data retrieval, LLM inference, and Git operations. It runs on a single-page frontend (vial.js, chatbot.js) hosted at webxos.netlify.app, with a Dockerized backend (/vial/) deployed via Vercel and Netlify. The system supports PostgreSQL, Milvus, Weaviate, pgvector, and FAISS for data retrieval, and LLaMA 3.3, Mistral, Gemma 2, Qwen, and Phi for LLMs. It integrates with github.com/webxos/webxos for Git operations.
Setup Instructions

Clone Repository:git clone https://github.com/webxos/webxos.git
cd webxos


Install Dependencies:
Python: pip install -r db/requirements.txt
Node.js: cd node && npm install


Set Environment Variables:Create a .env file:MONGO_URI=mongodb://localhost:27017
POSTGRES_URI=postgresql://user:password@localhost:5432/mcp_db
MILVUS_URI=localhost:19530
WEAVIATE_URI=http://localhost:8080
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET=your_jwt_secret
LLM_API_KEY=your_huggingface_api_key
OAUTH_CLIENT_ID=your_github_client_id
OAUTH_CLIENT_SECRET=your_github_client_secret
NOMIC_API_KEY=your_nomic_api_key


Run Services:docker-compose up -d


Deploy to Vercel:npm install -g vercel
vercel --prod


Access Frontend:
Vial: https://webxos.netlify.app/vial
Chatbot: https://webxos.netlify.app/chatbot
Streamlit Dashboard: http://localhost:8501



Git Commands
Use the following commands in vial.js or chatbot.js to manage the repository:

git clone: Clone the repository for training data.
git commit: Commit changes to training configurations.
git push: Push changes to the remote repository.
git pull: Pull latest updates from the repository.
git branch: Create or list branches for training variants.
git merge: Merge branches to consolidate training changes.
git checkout: Switch to a specific branch.
git diff: View changes between commits or branches.

API Endpoints

POST /v1/api/retrieve: Retrieve data from PostgreSQL, Milvus, Weaviate, pgvector, or FAISS.curl -X POST https://webxos.netlify.app/v1/api/retrieve \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "query": "test", "source": "postgres", "format": "json", "wallet": {}}'


POST /v1/api/llm: Call LLMs (LLaMA 3.3, Mistral, Gemma 2, Qwen, Phi).curl -X POST https://webxos.netlify.app/v1/api/llm \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "prompt": "Hello", "model": "llama3.3", "format": "json", "wallet": {}}'


POST /v1/api/git: Execute Git commands.curl -X POST https://webxos.netlify.app/v1/api/git \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "command": "git clone", "repo_url": "https://github.com/webxos/webxos.git", "wallet": {}}'



Troubleshooting

Check db/errorlog.md for logged errors.
Ensure SSL/TLS certificates are valid in nginx.conf.
Verify environment variables in .env.
Run tests: pytest vial/tests/ db/.
