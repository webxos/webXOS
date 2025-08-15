WebXOS Backend
Backend for the WebXOS Vial MCP Gateway, providing APIs for wallet management, authentication, quantum link, and vial data processing.
Features

Wallet Management: Handles .md wallet files with balance, reputation, and vial data.
Authentication: Supports OAuth client credentials and token-based authentication.
Quantum Link: Simulates quantum state synchronization.
Troubleshooting: Provides diagnostic endpoints.
API Configuration: Manages rate limits and settings.

Setup

Clone the repository:git clone https://github.com/webxos/webxos
cd webxos-backend


Install dependencies:npm install


Create a .env file (see .env artifact).
Add a sample_wallet.md file in the root directory (copy from vial_wallet_export_2025-08-15T19-31-10-169Z.md).
Run locally:npm start


Deploy to Netlify:
Push to GitHub.
In Netlify, set build command to npm install && npm start and publish directory to ..
Update API_BASE_URL in the frontend (index.html) to match your Netlify URL.



Endpoints

GET /v1/wallet: Returns wallet data (balance, reputation).
POST /v1/generate-credentials: Generates API key and secret.
POST /v1/oauth/token: Issues OAuth access token.
POST /v1/auth: Authenticates user.
POST /v1/void: Voids transactions.
GET /v1/troubleshoot: Runs diagnostics.
POST /v1/quantum-link: Syncs quantum state.
GET /v1/api-config: Returns API settings.
GET /v1/vial: Returns vial data (status, quantum state).

License
MIT
