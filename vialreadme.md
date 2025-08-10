Vial MCP Controller
The Vial MCP Controller is a decentralized, web-based master control panel (MCP) hosted at webxos.netlify.app/vial.html, managing a network of four PyTorch-based agentic AI models in a Dockerized /vial/ backend. The agents (MCP Host, Client, Server, Protocol) form an agentic network, validated via $WEBXOS proof-of-work SHA-256 hashes, designed for tasks like document search, email reading/sending, and web searching. The system supports recursive agent development through export/import of a single .md file, with SQLite for persistent storage and $WEBXOS rewards for training.
Features

Frontend: vial.html provides a responsive UI with real-time logging, stats, and controls.
Agentic Network: Four PyTorch agents (Host, Client, Server, Protocol) perform tasks (search_docs, read_emails, send_gmails, search_web) in parallel, validated by $WEBXOS hashes.
$WEBXOS Rewards: Training a vial (with uploaded file) awards 0.0001 $WEBXOS, validated via SHA-256, stored in the wallet.
Export/Import: Single .md file contains all agent code, wallet data, hashes, and session details, importable for further training.
SQLite Backend: Stores agent states, wallet data, and logs, accessible by the frontend.
Dockerized Deployment: Self-contained /vial/ folder for easy setup.
Open Source: Extendable at github.com/webxos/webxos.

Agent Roles

MCP Host: Initiates requests, handles user interaction (e.g., LLM/IDE), performs search_docs and read_emails.
MCP Client: Manages server connections, translates requests, handles send_gmails.
MCP Server: Exposes data sources (databases, APIs), performs search_web.
MCP Protocol: Defines communication standards, ensures secure data exchange.

Setup Instructions

Clone the Repository:git clone https://github.com/webxos/webxos.git
cd webxos


Install Docker:
Follow Docker's official installation guide.


Build and Run the Backend:cd vial
docker build -t vial-mcp-backend .
docker run -p 5000:5000 -d vial-mcp-backend


Serve the Frontend:
Host vial.html via Netlify or python -m http.server 8000.
Ensure /static/ contains dexie.min.js, redaxios.min.js, and icon.png.


CDN Dependencies:
Dexie: https://cdn.jsdelivr.net/npm/dexie@3.2.7/dist/dexie.min.js
Redaxios: https://cdn.jsdelivr.net/npm/redaxios@0.5.1/dist/redaxios.min.js
Download for /static/:curl -o static/dexie.min.js https://cdn.jsdelivr.net/npm/dexie@3.2.7/dist/dexie.min.js
curl -o static/redaxios.min.js https://cdn.jsdelivr.net/npm/redaxios@0.5.1/dist/redaxios.min.js





Usage

Access: Open https://webxos.netlify.app/vial.html.
Authenticate: Click "Authenticate" to initialize the agentic network and $WEBXOS wallet.
Train Agents:
Upload a .py, .js, .txt, or .md file with training data (e.g., custom prompt for search_docs).
Click "Train Vials" to train all agents, earning 0.0001 $WEBXOS per vial (total 0.0004 per session).
Agents validate each other via $WEBXOS SHA-256 hashes.


Export:
Click "Export" to download a .md file with:
Agent code, status, $WEBXOS hash, wallet data.
Session timestamp, network ID, and total $WEBXOS balance.
Instructions for reuse in other projects.




Import:
Upload the exported .md via the "Upload" button to resume training.
The system parses the .md to restore agent states and wallet.


Troubleshoot: Diagnose server/database issues with the "Troubleshoot" button.
Void: Reset the system with the "Void" button.

Agent Details

Structure: Each agent is a PyTorch nn.Module with task-specific logic (search_docs, read_emails, send_gmails, search_web).
Training: Accepts uploaded data or default code, trains in parallel, updates $WEBXOS balance.
Coordination: Uses PyTorch tensors for quantum simulation-inspired positioning, validated by $WEBXOS hashes.
Export/Import: .md file contains all agent data, parseable for reimport.

$WEBXOS Integration

Proof-of-Work: Each training session generates a SHA-256 hash linking agents to the wallet address.
Rewards: 0.0001 $WEBXOS per vial per training session, stored in SQLite and exported in .md.
Wallet: Tracks session balance, distributed equally among agents, supports future Stripe cash-outs (not implemented).

Deployment Diagram
+-------------------+       +-------------------+
|   vial.html       |<----->|   /vial/ Backend  |
| (Netlify/Static)  |       | (Docker Container)|
| - Dexie           |       | - server.py       |
| - Redaxios        |       | - vial_manager.py |
| - UI Controls     |       | - agent_*.py      |
|                   |       | - SQLite DB       |
+-------------------+       +-------------------+
           |                        |
           v                        v
+-------------------+       +-------------------+
|   /static/        |       |   GitHub Repo     |
| - dexie.min.js    |       | - webxos/webxos   |
| - redaxios.min.js |       | - Push Updates    |
| - icon.png        |       +-------------------+
+-------------------+

Development

Extend Agents: Modify agent_*.py to enhance task logic or add data sources.
Import/Export: Use exported .md files to build new agents externally, then reimport.
Contribute: Fork github.com/webxos/webxos, submit PRs.
Error Logging: View logs in vial.html console, stored in SQLite or localStorage.

License
MIT License. See LICENSE.
