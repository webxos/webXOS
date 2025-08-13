Vial MCP Controller - Vial Module README
Overview
The vial module is the core backend for the Vial MCP Controller, managing AI agents (vials), $WEBXOS wallet operations, and quantum-inspired task processing. It integrates with MongoDB, SQLite, and Redis for data persistence and synchronization, maintaining compatibility with the WebXOS wallet export format and supporting exactly four vials.
Features

Vial Management: Handles four AI agents (vial1-4) with training, status monitoring, and export/import functionality.
Wallet Integration: Manages $WEBXOS transactions and balance tracking via webxos_wallet.py.
Quantum Simulation: Simulates task queuing using Qiskit in quantum_simulator.py.
Authentication: JWT-based security via auth_manager.py.
Offline Support: Uses SQLite (database.sqlite) and Dexie.js for local storage.
Error Logging: Logs errors to errorlog.md for debugging.

Setup Instructions

Prerequisites:

Python 3.8+
MongoDB
Redis 7.0
Dependencies listed in /db/requirements.txt


Installation:
cd vial-mcp-project
pip install -r db/requirements.txt
python db/mcp_db_init.py


Running the Server:
uvicorn vial.unified_server:app --host 0.0.0.0 --port 8000


Testing:
pytest vial/tests/



Usage

Wallet Operations: Use /api/wallet and /api/wallet/cashout endpoints to manage $WEBXOS balances.
Vial Management: Use /api/vials and /api/vial/update to manage vial data.
Import/Export: Import wallet exports via /api/import, ensuring four vials and valid hashes.
Frontend: Access vial.html for the retro terminal UI (neon green on black, Courier New font).

File Structure

auth_manager.py: JWT authentication.
client.py: Client-side API interactions.
export_manager.py: Markdown export generation and validation.
langchain_agent.py: Query enhancement and vial training.
network_sync.py: Synchronizes SQLite and MongoDB.
quantum_simulator.py: Quantum-inspired task processing.
unified_server.py: FastAPI backend for core endpoints.
vial_manager.py: Manages vial data and validation.
webxos_wallet.py: $WEBXOS wallet operations.
database.sqlite: Local wallet storage.
errorlog.md: Error logs.
/prompts/: LangChain prompt templates.
/static/: Frontend assets (Dexie.js, Redaxios).
/tests/: Unit tests for all components.

Notes

The chatbot folder is isolated and not modified.
Wallet exports must follow the format in vial_wallet_export_*.md with four vials.
Ensure .env includes WEB3_PROVIDER, JWT_SECRET, MONGO_URL, and REDIS_URL.
