Vial MCP Controller
Updated on August 10, 2025.
A remote controller for managing a 4x agentic quantum simulated network, integrated with $WEBXOS wallet for decentralized payouts and app development. Supports LangChain for LLM integration and exports/imports .md files.
Features

4x Agentic Network: Four vial agents with quantum simulations and LangChain support.
$WEBXOS Wallet: Earn and manage $WEBXOS tokens, exportable to .md.
PyTorch Integration: Train and export models as .md files.
LangChain Compatibility: Supports LangGraph for future LLM integrations.
Offline Fallback: Uses Dexie.js for local storage of exports.
vial.html Controller: Browser-based UI for controlling the /vial/ backend.

Setup

Clone the Repository:git clone https://github.com/your-username/vial-mcp-project.git
cd vial-mcp-project


Install Dependencies:pip install -r vial/requirements.txt


Run Docker Container:docker build -t vial-mcp -f vial/Dockerfile vial
docker run -d -p 5000:5000 --rm -v /tmp:/data/vial_results vial-mcp


Access Controller:Open vial.html in a browser to control the backend at http://localhost:5000.

Usage

Authenticate: Use the "Authenticate" button in vial.html to connect.
Train Vials: Upload .md files with JSON-like data to train agents.
Export: Export trained models and wallet data as .md files.
Offline Mode: Exports are saved locally via Dexie.js if offline.

Contributing
Submit issues to errorlog.md and pull requests to enhance functionality.
License
MIT License. See LICENSE for details.
[xaiartifact: v1.7]
