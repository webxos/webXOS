Database Documentation
Overview
The Vial MCP Controller uses MongoDB and SQLite for data persistence, managing vials and $WEBXOS wallet data.
MongoDB

Database: mcp_db
Collections:
users: Stores user IDs and API keys.
wallet: Tracks $WEBXOS balances and transactions.
vials: Manages vial data (status, scripts, wallet info).
errors: Logs errors with timestamps.


Indexes:
userId (unique) on users.
userId (unique) on wallet.
id (unique) on vials.
timestamp on errors.



SQLite

Database: vial/database.sqlite
Tables:
wallets: Stores user IDs and balances.
vials: Mirrors MongoDB vials collection for offline sync.



Setup

Run mcp_db_init.py to initialize MongoDB collections.
Ensure database.sqlite exists in /vial for wallet operations.
Use network_sync.py to synchronize MongoDB and SQLite.

Error Handling

Errors are logged to vial/errorlog.md.
Check MongoDB connection in unified_server.py.
