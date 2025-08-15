WebXOS Backend Project Guide
This guide provides detailed instructions for developers to build and extend the WebXOS Backend, addressing issues like those encountered in the error logs (JSON parse errors due to HTML responses).
Project Overview
The WebXOS Backend powers the Vial MCP Gateway, handling wallet management, authentication, quantum synchronization, and vial data processing. It supports .md wallet files (e.g., vial_wallet_export_2025-08-15T19-31-10-169Z.md) and resolves errors from missing or misconfigured API endpoints.
Error Resolution
The error logs show HTML responses (content-type: text/html) with a 200 status code, indicating a misconfigured backend. The provided files fix this by:

Implementing all required endpoints (/wallet, /generate-credentials, /oauth/token, /auth, /void, /troubleshoot, /quantum-link, /api-config, /vial).
Ensuring JSON responses with proper content-type: application/json.
Adding error handling and logging.

Setup Instructions

Clone and Install:git clone https://github.com/webxos/webxos
cd webxos-backend
npm install


Configure Environment:
Create .env (see .env artifact).
Add sample_wallet.md in the root directory (copy from vial_wallet_export_2025-08-15T19-31-10-169Z.md).


Run Locally:npm run dev


Test Endpoints:
Use Postman or curl, e.g.:curl http://localhost:3000/v1/wallet

Expected: {"balance":0,"reputation":0}


Deploy to Netlify:
Push to GitHub.
In Netlify, set build command to npm install && npm start and publish directory to ..
Update API_BASE_URL in the frontend (index.html).



Extending the Backend

Wallet Parsing: Modify src/services/walletService.js to support additional .md fields (e.g., Training Data).
Authentication: Enhance src/middleware/auth.js with JWT or database-backed token validation.
Database: Replace file-based wallet storage with a database (e.g., MongoDB).
Testing: Add unit tests using Jest in a tests/ directory.

Troubleshooting

JSON Parse Errors: Ensure all routes return application/json. Check Netlify logs for deployment issues.
CORS Issues: Verify cors middleware is enabled.
File Parsing: Ensure sample_wallet.md matches the expected .md format.

License
MIT
