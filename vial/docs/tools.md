Vial MCP Controller Tools
Overview
The Vial MCP Controller provides extensible tools and agents for data retrieval, LLM inference, and Git operations. Tools are configured in resources/mcp_tools.json and integrated with the API Gateway (vial/unified_server.py).
Available Tools

Sample Tool (sample_tool.py):
Description: A customizable tool that echoes input with a timestamp for testing purposes.
Endpoint: /v1/api/sample_tool
Usage: Send a POST request with {"user_id": "user123", "input": "test"}.
Example:curl -X POST https://webxos.netlify.app/v1/api/sample_tool \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "input": "test"}'





Agent Tools

Nomic Search (agent1.py):
Description: Vector-based search using Nomic embeddings.
Endpoint: /v1/api/nomic_search
Config: resources/mcp_tools.json (nomic_search).
Example:curl -X POST https://webxos.netlify.app/v1/api/nomic_search \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "query": "test", "limit": 5}'




CogniTALLMware Search (agent2.py):
Description: Vector-based search using CogniTALLMware embeddings.
Endpoint: /v1/api/cognitallmware_search
Config: resources/mcp_tools.json (cognitallmware_search).


LLMware Search (agent3.py):
Description: Vector-based search using LLMware embeddings.
Endpoint: /v1/api/llmware_search
Config: resources/mcp_tools.json (llmware_search).


Jina AI Search (agent4.py):
Description: Vector-based search using Jina AI embeddings.
Endpoint: /v1/api/jinaai_search
Config: resources/mcp_tools.json (jinaai_search).



Git Operations

Tool: Git Operations
Description: Execute Git commands for repository management (github.com/webxos/webxos).
Endpoint: /v1/api/git
Supported Commands: git clone, git commit, git push, git pull, git branch, git merge, git checkout, git diff, git log, git status.
Config: resources/mcp_tools.json (git_operations).
Example:curl -X POST https://webxos.netlify.app/v1/api/git \
     -H "Authorization: Bearer <your_token>" \
     -d '{"user_id": "user123", "command": "git status", "repo_url": "https://github.com/webxos/webxos.git"}'



Adding New Tools

Create a Python script in /tools/ following sample_tool.py structure.
Update resources/mcp_tools.json with tool details (name, endpoint, config).
Register the endpoint in vial/unified_server.py.
Test with curl or frontend (vial.js, chatbot.js).
