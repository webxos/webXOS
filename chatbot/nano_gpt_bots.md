Nano GPT Bots Documentation
Overview
The nano_gpt_bots.md file provides documentation for the chatbot module of the Vial MCP Controller project, focusing on the integration of nano-GPT models for conversational AI. This module supports the primary (chatbot.html) and secondary (chatbot2.html) chatbot UIs, interfacing with the backend (server.py) and service worker (sw.js).
Components

chatbot.html: Primary chatbot UI, accessible from the root directory, providing a retro terminal interface.
chatbot2.html: Secondary chatbot UI with a simplified interface for quick interactions.
server.py: FastAPI backend handling chatbot queries, integrating with nano-GPT models.
site_index.json: Search index for chatbot responses, enabling fuzzy search via /static/fuse.min.js.
sw.js: Service worker for caching chatbot responses, ensuring offline functionality.

Integration

Nano-GPT Models: The chatbot uses lightweight GPT models (distilgpt2) for generating responses, configured in server.py.
Wallet System: Queries update the wallet via webxos_wallet.py, incrementing webxos balance by 0.0001.
Inception Gateway: Queries are routed through endpoints defined in db/library_config.json.
Frontend: Both UIs (chatbot.html, chatbot2.html) use /static/style.css and /vial/static/redaxios.min.js for styling and HTTP requests.

Setup

Install dependencies: pip install -r db/requirements.txt (includes transformers for nano-GPT).
Start the chatbot server: python chatbot/server.py.
Access UIs at /chatbot.html or /chatbot/chatbot2.html via webxos.netlify.app.

Error Handling
Errors are logged to db/errorlog.md with timestamps to prevent redundancy. Check this file for issues related to chatbot operations.
For API details, see /docs/api.markdown.
