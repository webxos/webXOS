Nano GPT Bots Documentation
Overview
The Vial MCP chatbot (chatbot.html) integrates with the backend to provide an interactive interface for quantum processing, note management, and resource retrieval. It leverages four JavaScript agents (agent1.js, agent2.js, agent3.js, agent4.js) to communicate with the OAuth-embedded MCP server.
Features

Authentication: Uses OAuth via /api/auth/login and /api/auth/refresh endpoints, requiring API key and wallet ID.
Quantum Processing: Sends prompts to /api/quantum/link for simulated quantum state processing.
Note Management: Adds and reads notes via /api/notes/add and /api/notes/read, stored in SQLite and /app/notes/.
Resource Retrieval: Fetches latest notes as resources via /api/resources/latest.
Offline Support: Uses Dexie.js for client-side logging and a service worker (sw.js) for caching.

Integration

Frontend: chatbot.html mirrors vial.html but focuses on conversational interactions.
Agents:
agent1.js: Handles OAuth authentication.
agent2.js: Manages quantum processing.
agent3.js: Handles note creation and retrieval.
agent4.js: Retrieves resources.


Backend: Integrates with mcp_server_auth.py, mcp_server_quantum.py, mcp_server_notes.py, and mcp_server_resources.py.
Security: All requests use HTTPS and OAuth tokens, with wallet-based access control.

Usage

Open chatbot.html in a browser.
Enter API key (e.g., api-a24cb96b-96cd-488d-a013-91cb8edbbe68) and wallet ID (e.g., wallet_123).
Click "Authenticate" to obtain an OAuth token.
Use the interface to send prompts, add notes, or retrieve resources.
Logs are displayed in the console div and stored in Dexie.

Development Notes

Prompt Automation: Prompts are sourced from prompt_pool.json via base_prompt.py.
Error Handling: Errors are logged to the console and Dexie, with tracebacks in errorlog.md.
Testing: Use test_mcp_server_*.py to validate backend endpoints.
Future Improvements:
Add real-time chat streaming.
Enhance prompt automation with dynamic generation.


