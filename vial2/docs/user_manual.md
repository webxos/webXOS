Vial2 MCP User Manual
Overview
The Vial2 MCP Controller is a standalone HTML5 PWA for managing AI agents in the WebXOS network via vial2.html. It integrates with a FastAPI backend, Neon PostgreSQL, Stack Auth OAuth2.0, and PyTorch for agent training.
Getting Started

Setup:

Clone the repository: git clone https://github.com/webxos/vial2-mcp
Install dependencies: pip install -r vial2/requirements.txt
Configure .env with Neon DB and Stack Auth credentials.
Run migrations: bash vial2/scripts/migrate.sh
Start server: python vial2/main.py
Open vial2.html in a browser.


Authentication:

Click "Authenticate" button to initiate OAuth2.0 flow with Stack Auth.
Verify wallet address and access token are displayed.


Console Commands:

Use the console in vial2.html to execute commands:
/prompt <vial> <text>: Send prompt to a vial.
/task <vial> <task>: Assign a task.
/config <vial> <key> <value>: Configure vial settings.
/status: Check system status.
/git <command>: Run Git commands (e.g., status, pull).
/configure: Configure compute resources.
/refresh_configuration: Refresh system settings.
/terminate_fast: Stop running vials.
/help: Display command help.




Wallet Operations:

Import: Use "Import" button to load .md wallet files.
Export: Use "Export" button to save wallet data.
Merge: Combine wallets via API endpoint /mcp/api/wallet_merge.


Quantum Link:

Use "Quantum Link" button to initiate PyTorch model training.
Monitor training via /mcp/api/status.



Offline Mode

Install vial2.html as a PWA for offline access.
Local API simulation and SQLite error logging are supported.

Troubleshooting

See docs/troubleshooting.md for common issues and solutions.
Check logs via /mcp/api/logs or error_log.db.

Support

GitHub Issues: webxos/vial2-mcp
Docs: Neon, Stack Auth, Anthropic MCP

xAI Artifact Tags: #vial2 #docs #user_manual #neon_mcp
