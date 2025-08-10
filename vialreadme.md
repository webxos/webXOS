Vial MCP Controller
Updated on August 10, 2025.
A remote controller for managing a 4x agentic quantum simulated network, integrated with $WEBXOS wallet for decentralized payouts and app development. The system uses PyTorch for exporting trained models and supports offline fallback.
Features

4x Agentic Network: Four pre-configured vial agents running quantum simulations, enhanced with prototype agents.
Prototype Agents: agent1.py to agent4.py as baseline templates for modification and coordination with vial1-4.
$WEBXOS Wallet: Earn and manage $WEBXOS tokens, exportable to Stripe for cashouts.
PyTorch Integration: Export trained models as .md files with wallet data.
Offline Fallback: Maintains export functionality without internet via local storage.
vial.html Controller: Browser-based UI for controlling the /vial/ backend.

Setup

Clone the Repository:git clone https://github.com/your-username/vial-mcp-project.git
cd vial-mcp-project


Install Dependencies:pip install -r vial/requirements.txt


Run Docker Container:docker build -t vial-mcp -f vial/Dockerfile vial
docker run -d -p 5000:5000 --rm -v /tmp:/data/vial_results vial-mcp


Access Controller:Open vial.html in a browser to control the backend at http://localhost:5000.

Usage

Authenticate: Use the "Authenticate" button in vial.html to connect to the backend.
Train Vials: Upload .py or .md files to train the 4x agents, enhanced by prototype agents in agent1-4.py.
Export: Export trained models and $WEBXOS wallet data as .md files.
Offline Mode: If the internet is lost, exports are saved locally via Dexie.js.

Contributing
Submit issues to errorlog.md and pull requests to enhance functionality.
License
MIT License. See LICENSE for details.
