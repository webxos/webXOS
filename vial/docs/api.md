Vial MCP Controller API Documentation
Overview
The Vial MCP Controller API manages AI agents (vials) and $WEBXOS wallet operations. It supports authentication, vial management, and wallet transactions.
Endpoints
POST /api/auth

Description: Authenticate user and generate JWT token.
Request Body: {"userId": "string"}
Response: {"apiKey": "string"}
Errors: 500 (Server error)

GET /api/vials

Description: Retrieve all vials.
Security: Bearer token
Response: {"agents": {}}
Errors: 401 (Unauthorized), 500 (Server error)

POST /api/wallet

Description: Update wallet balance.
Security: Bearer token
Request Body: {"transaction": {"amount": number}, "wallet": {"target_address": "string"}}
Response: {"status": "success"}
Errors: 401 (Unauthorized), 500 (Server error)

POST /api/wallet/cashout

Description: Cash out $WEBXOS.
Security: Bearer token
Request Body: {"transaction": {"amount": number}, "wallet": {"target_address": "string"}}
Response: {"status": "success"}
Errors: 401 (Unauthorized), 500 (Server error)

POST /api/import

Description: Import wallet and vial export.
Security: Bearer token
Request Body: Multipart form-data with file
Response: {"agents": {}}
Errors: 400 (Invalid format), 401 (Unauthorized), 500 (Server error)

Authentication

Uses JWT tokens generated via /api/auth.
Include token in Authorization: Bearer <token> header.
