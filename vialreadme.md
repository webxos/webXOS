Vial MCP Controller
[xaiartifact: v1.6]
A remote controller for managing a 4x agentic quantum-simulated network with $WEBXOS wallet integration. The backend uses aiohttp for HTTP requests and sqlite3 for data persistence, replacing Redaxios and Dexie for a leaner client-side experience.
Repository Structure
/vial/
├── server.py                 # Aiohttp server with API endpoints
├── vial_manager.py           # Manages vial agents and database
├── requirements.txt          # Dependencies: aiohttp, torch, treeshaker
├── Dockerfile                # Docker setup for backend
├── static/
│   ├── icon.png             # Favicon
├── errorlog.md              # Error log
├── README.md                # This file
├── treeshaker.cfg            # Treeshaker config
└── vial.html                # Frontend controller

Setup

Clone Repository:git clone <repo-url>
cd vial


Install Dependencies:pip install -r requirements.txt


Tree-Shake Backend:treeshaker --config treeshaker.cfg

Ensure treeshaker.cfg specifies server.py and vial_manager.py as targets.
Build and Run Docker:docker build -t vial-mcp .
docker run -p 8080:8080 vial-mcp


Access Controller:Open http://localhost:8080/vial.html.

Usage

Authenticate: Click "Authenticate" to initialize a network ID and wallet.
Train Vials: Upload a .py, .js, .txt, or .md file to train 4 vial agents.
Export: Download a .md file with vial and wallet data.
Void: Reset all data.
Troubleshoot: Check server status.
Offline Mode: Functions without internet, storing data in memory.

Troubleshooting

HTTP 404 Errors:
Ensure server.py is running: python server.py or verify Docker container.
Check server.log for errors (e.g., route registration, database issues).
Verify port 8080 is open: netstat -tuln | grep 8080 or docker ps.
Confirm client serverUrl in vial.html is /api/mcp.
Verify endpoint paths in vial.html API calls match backend routes (e.g., /api/mcp/auth, /api/mcp/health).
Test server health: curl http://localhost:8080/api/mcp/health.
Restart Docker: docker stop <container-id> && docker run -p 8080:8080 vial-mcp.


Train Error: No file selected:
Select a file via the "Upload" button before clicking "Train Vials".


Server Not Responding:
Check server.log for startup errors or missing dependencies.
Ensure /uploads/ directory exists: mkdir -p /uploads.
Verify vial.db is writable: chmod 666 vial.db.



Features

Backend-driven with aiohttp and sqlite3 for HTTP and storage.
Tree-shaked with treeshaker to minimize backend code.
Supports PyTorch-based agent training.
$WEBXOS wallet integration for future Stripe cash-outs.

Notes

Ensure /uploads/ exists for file uploads.
Update errorlog.md with any runtime issues.
Backend runs on port 8080; adjust Dockerfile if needed.
