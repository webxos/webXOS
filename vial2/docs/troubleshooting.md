Troubleshooting Guide
Common Issues
1. Authentication Failure

Symptoms: OAuth2.0 flow fails, "Invalid token" error.
Solution:
Verify STACK_AUTH_CLIENT_ID and STACK_AUTH_CLIENT_SECRET in .env.
Check redirect URI matches http://webxos.netlify.app or http://localhost:8000.
Ensure Stack Auth JWKS URL is accessible: https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/.well-known/jwks.json.



2. Database Connection Errors

Symptoms: "Connection refused" or "SSL error" when connecting to Neon DB.
Solution:
Confirm DATABASE_URL in .env matches Neon DB connection string.
Run psql $DATABASE_URL to test connection.
Check Neon Console for branch status.



3. API Rate Limiting

Symptoms: HTTP 429 "Rate limit exceeded".
Solution:
Wait 60 seconds and retry.
Check RateLimiter settings in security/rate_limiter.py for max_requests and window_seconds.



4. Git Command Errors

Symptoms: "Unsupported Git command" or command failure in console.
Solution:
Ensure commands are limited to status, pull, commit, push.
Verify Git is installed and configured in the environment.



5. PyTorch Model Training Issues

Symptoms: Model training fails or returns unexpected results.
Solution:
Check torch.__version__ for compatibility (python -c "import torch; print(torch.__version__)").
Verify GPU/CPU detection in pytorch/quantum_link.py.
Ensure model data is correctly formatted in requests.



Logs

Check SQLite error logs in error_log.db for detailed error messages.
Use /mcp/api/logs endpoint to retrieve aggregated logs.

Support

Report issues: GitHub Issues
Neon DB: Neon Docs
Stack Auth: Stack Auth Docs
Anthropic MCP: Anthropic Docs

xAI Artifact Tags: #vial2 #docs #troubleshooting #neon_mcp
