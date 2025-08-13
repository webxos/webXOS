Vial MCP Controller API Documentation
Overview
The Vial MCP Controller API provides endpoints for managing AI agents (vials), $WEBXOS wallet transactions, and authentication. All endpoints require JWT authentication unless specified.
Base URL
http://localhost:8000/api
Authentication

Endpoint: /auth
Method: POST
Request Body:{
  "userId": "string"
}


Response:{
  "apiKey": "string"
}


Description: Generates a JWT token for the user.

Wallet Operations

Endpoint: /wallet

Method: POST

Headers: Authorization: Bearer <apiKey>

Request Body:
{
  "transaction": {"amount": "float"},
  "wallet": {"target_address": "string"}
}


Response:
{"status": "success"}


Description: Updates wallet balance and logs transactions.

Endpoint: /wallet/cashout

Method: POST

Headers: Authorization: Bearer <apiKey>

Request Body:
{
  "transaction": {"amount": "float"},
  "wallet": {"target_address": "string"}
}


Response:
{"status": "success"}


Description: Initiates a cashout transaction.


Vial Operations

Endpoint: /vials

Method: GET

Headers: Authorization: Bearer <apiKey>

Response:
{
  "agents": {
    "vial1": {"status": "string", "wallet_balance": "float", "wallet_address": "string", "wallet_hash": "string", "script": "string"},
    ...
  }
}


Description: Retrieves all vial data.

Endpoint: /vial/update

Method: POST

Headers: Authorization: Bearer <apiKey>

Request Body:
{
  "vial_id": "string",
  "data": {"status": "string", "wallet_balance": "float", "wallet_address": "string", "wallet_hash": "string", "script": "string"}
}


Response:
{"status": "success"}


Description: Updates a specific vial's data.

Endpoint: /vial/wallet

Method: GET

Headers: Authorization: Bearer <apiKey>

Query Parameters: vial_id=string

Response:
{
  "vial_id": "string",
  "balance": "float"
}


Description: Retrieves wallet balance for a specific vial.


Import/Export

Endpoint: /import
Method: POST
Headers: Authorization: Bearer <apiKey>
Request Body: Multipart form-data with a .md file
Response:{
  "agents": {
    "vial1": {"status": "string", "wallet_balance": "float", "wallet_address": "string", "wallet_hash": "string", "script": "string"},
    ...
  }
}


Description: Imports a wallet and vial export file, validating the format and updating the database.

Error Handling
All endpoints return standard HTTP error codes (e.g., 401 for unauthorized, 500 for server errors) with detailed error messages logged to vial/errorlog.md.
