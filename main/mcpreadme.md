Neon MCP Controller
Overview
Neon MCP (Machine Control Protocol) is a decentralized platform for managing up to four AI agents (vials) with $WEBXOS token transactions secured by a proof-of-work blockchain. The frontend provides a retro terminal interface for authentication, training, exporting/importing vials, and API access. The backend uses FastAPI, MongoDB, PyTorch, TensorFlow, and DSPy, deployable on Netlify with GitHub integration.
Features

Manage four vials with individual $WEBXOS balances and training states.
Proof-of-work blockchain for secure $WEBXOS transactions.
Export/import vial configurations as Markdown files.
OAuth-based authentication with API credential management.
Deployable on Netlify with seamless GitHub integration.

Prerequisites

Python 3.9+
MongoDB
Node.js (for Netlify CLI)
Git
Netlify account
GitHub repository

Setup

Clone Repository:
git clone https://github.com/your-username/neon-mcp.git
cd neon-mcp


Install Dependencies:
pip install -r requirements.txt


Configure Environment:Create a .env file:
cp .env.example .env

Edit .env with:
JWT_SECRET=secret_key_123_change_in_production
MONGO_URL=mongodb://localhost:27017
OPENAI_API_KEY=your-openai-key


Start MongoDB:
mongod --dbpath ./data/db --fork --logpath ./logs/mongodb.log


Run Locally:
uvicorn main.api.main:app --host 0.0.0.0 --port 8000 --reload



Deployment on Netlify

Push to GitHub:
git add .
git commit -m "Initial commit"
git push origin main


Configure Netlify:

Link your GitHub repository in Netlify.
Set build command: pip install -r requirements.txt && uvicorn main.api.main:app --host 0.0.0.0 --port $PORT
Set publish directory: static
Add environment variables in Netlify dashboard: JWT_SECRET, MONGO_URL, OPENAI_API_KEY.


Deploy:
netlify deploy --prod



API Endpoints

GET /api/v1/health: Check server status and wallet balance.
POST /api/v1/oauth/token: Authenticate and get JWT.
POST /api/v1/train/{vial_id}: Train a vial.
POST /api/v1/void: Reset all vials and wallet.
GET /api/v1/troubleshoot: Diagnose system issues.
POST /api/v1/quantum_link: Train all vials with PoW rewards.
GET /api/v1/credentials: Get API credentials.

Project Structure
main/
├── api/
│   ├── main.py          # FastAPI entry point
│   ├── mcp/
│   │   ├── server.py    # Core MCP logic
│   │   ├── blockchain.py # Blockchain for $WEBXOS
│   ├── utils/
│   │   ├── logging.py   # Logging setup
│   └── routes/
├── static/
│   └── index.html       # Frontend interface
├── .env                 # Environment variables
├── netlify.toml         # Netlify configuration
└── README.md            # Project documentation

Troubleshooting

NetworkError: Ensure backend is running (./start_server.sh) or check .env configuration.
Invalid JSON: Verify endpoint responses in /main/api/main.py.
CORS Issues: Check CORS settings in main.py.

License
MIT License. $WEBXOS token is in development; no liability for token loss.
