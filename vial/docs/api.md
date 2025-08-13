Vial MCP API Documentation
Overview
The Vial MCP API provides endpoints for managing AI vials, wallets, authentication, and blockchain transactions. The API runs on http://localhost:8000/api by default.
Endpoints
Health Check

GET /api/health
Description: Checks the health of the backend and its services.
Response:{
  "status": "healthy",
  "mongo": true,
  "version": "2.8",
  "services": ["auth", "wallet", "vials"]
}


Errors:
503: Service unavailable, with error details.





Authentication

POST /api/auth/login

Description: Authenticates a user and returns an API key.
Request:{"userId": "user-123"}


Response:{
  "apiKey": "JWT-uuid",
  "walletAddress": "hash",
  "walletHash": "hash"
}


Errors:
400: Missing userId.
500: Server error.




POST /api/auth/api-key/generate

Description: Generates a new API key for a user.
Request:{"userId": "user-123"}


Response:{
  "apiKey": "JWT-uuid",
  "walletAddress": "hash",
  "walletHash": "hash"
}


Errors:
400: Missing userId.
500: Server error.





Vial Operations

POST /api/vials/{vial_id}/prompt

Description: Sends a prompt to a specific vial.
Request:{
  "vialId": "vial1",
  "prompt": "Train model",
  "blockHash": "hash"
}


Response:{"response": "Prompt processed for vial1"}


Errors:
400: Invalid vial ID.
500: Server error.




POST /api/vials/{vial_id}/task

Description: Assigns a task to a specific vial.
Request:{
  "vialId": "vial1",
  "task": "Process data",
  "blockHash": "hash"
}


Response:{"status": "Task assigned to vial1"}


Errors:
400: Invalid vial ID.
500: Server error.




PUT /api/vials/{vial_id}/config

Description: Updates configuration for a specific vial.
Request:{
  "vialId": "vial1",
  "key": "model",
  "value": "gpt-3",
  "blockHash": "hash"
}


Response:{"status": "Config updated for vial1"}


Errors:
400: Invalid vial ID.
500: Server error.




DELETE /api/vials/void

Description: Resets all vials to stopped state.
Response:{"status": "All vials reset"}


Errors:
500: Server error.





Wallet Operations

POST /api/wallet/create

Description: Creates a new wallet for a user.
Request:{
  "userId": "user-123",
  "address": "wallet-123",
  "balance": 0,
  "hash": "hash",
  "webxos": 0.0
}


Response:{
  "status": "Wallet created",
  "address": "wallet-123"
}


Errors:
500: Server error.




POST /api/wallet/import

Description: Imports a wallet for a user.
Request:{
  "userId": "user-123",
  "address": "wallet-123",
  "hash": "hash",
  "webxos": 0.0
}


Response:{"status": "Wallet imported"}


Errors:
500: Server error.




POST /api/wallet/transaction

Description: Records a wallet transaction.
Request:{
  "userId": "user-123",
  "type": "transaction"
}


Response:{"status": "Transaction recorded"}


Errors:
400: No wallet found.
500: Server error.





Quantum Link

POST /api/quantum/link
Description: Establishes a quantum link for vials.
Request:{"vials": ["vial1", "vial2"]}


Response:{
  "statuses": ["running", "running"],
  "latencies": [50, 60]
}


Errors:
500: Server error.





Blockchain

POST /api/blockchain/transaction
Description: Records a blockchain transaction.
Request:{
  "type": "command",
  "data": {},
  "timestamp": "2025-08-13T17:00:00Z",
  "hash": "hash"
}


Response:{"status": "Transaction recorded"}


Errors:
500: Server error.





Error Logging

POST /api/log_error
Description: Logs an error from the frontend.
Request:{
  "error": "Error message",
  "endpoint": "/api/health",
  "timestamp": "2025-08-13T17:00:00Z",
  "source": "frontend",
  "rawResponse": "Response text"
}


Response:{"status": "logged"}


Errors:
500: Server error.




