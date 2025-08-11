# Vial MCP Controller

![Vial MCP Logo](static/icon.png)

A lightweight, agentic MCP server for managing a 4x quantum-simulated network with $WEBXOS wallet integration. Built with FastAPI and FastMCP, it supports LangChain, Qiskit, Web3.py, and SQLite for a minimal, scalable setup. Deployed at [webxos.netlify.app](https://webxos.netlify.app) and maintained at [github.com/webxos/webxos](https://github.com/webxos/webxos). Follow us on [X @webxos](https://x.com/webxos).

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Rebuild Instructions](#rebuild-instructions)
- [Usage](#usage)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **4x Agentic Network**: Four PyTorch-based agents with Qiskit quantum simulation.
- **$WEBXOS Wallet**: Decentralized payouts via Web3.py, persisted in SQLite.
- **FastAPI + FastMCP**: High-performance APIs exposed as MCP tools/resources.
- **LangChain Integration**: NanoGPT with reusable prompt templates.
- **Real-Time Streaming**: Server-Sent Events (SSE) for live updates.
- **Security**: JWT authentication, HTTPS, and offline fallback with Dexie.js.
- **Git Terminal**: API-driven git interactions in `vial.html`.
- **Exports**: Markdown exports with WEBXOS Tokenization Tag.

## Project Structure
```
vial-mcp-project/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions for CI/CD
├── vial/
│   ├── server.py                   # FastAPI + FastMCP server
│   ├── vial_manager.py             # Manages 4x agents and tools
│   ├── quantum_simulator.py        # Qiskit-based quantum simulation
│   ├── webxos_wallet.py            # $WEBXOS wallet with Web3.py, SQLite
│   ├── auth_manager.py             # JWT authentication
│   ├── export_manager.py           # Markdown exports
│   ├── langchain_agent.py          # LangChain with NanoGPT
│   ├── client.py                   # MCP client for testing
│   ├── tools/
│   │   └── base_tool.py           # Sample MCP tool
│   ├── prompts/
│   │   └── base_prompt.py         # Sample prompt template
│   ├── agents/
│   │   ├── agent1.py              # Agent 1 with PyTorch
│   │   ├── agent2.py              # Agent 2 with PyTorch
│   │   ├── agent3.py              # Agent 3 with PyTorch
│   │   └── agent4.py              # Agent 4 with PyTorch
│   ├── tests/
│   │   ├── test_server.py         # Server tests
│   │   ├── test_vial_manager.py   # Vial manager tests
│   │   ├── test_wallet.py         # Wallet tests
│   │   ├── test_quantum_simulator.py # Quantum simulator tests
│   │   ├── test_auth_manager.py   # Auth manager tests
│   │   ├── test_export_manager.py # Export manager tests
│   │   └── test_langchain_agent.py # LangChain agent tests
│   ├── requirements.txt            # Python dependencies
│   ├── .env                       # Environment variables
│   ├── openapi.yaml               # API schema
│   └── Dockerfile                 # Docker configuration
├── static/
│   ├── icon.png                   # Favicon (CDN-downloaded)
│   ├── dexie.min.js               # Dexie.js for offline storage (CDN-downloaded)
│   └── redaxios.min.js            # Redaxios for HTTP requests (CDN-downloaded)
├── _docs/
│   ├── tools.md                   # Tool documentation
│   └── resources.md               # Resource documentation
├── README.md                      # Project overview
├── errorlog.md                    # Error tracking
├── llms.txt                       # LLM-friendly docs
├── vial.html                      # Frontend controller
├── netlify.toml                   # Netlify configuration
├── docker-compose.yaml            # Docker orchestration
├── wallet.db                      # SQLite database (auto-generated)
└── LICENSE                        # MIT License
```

## Prerequisites
- Git
- Python 3.11
- Docker and Docker Compose
- Node.js and Yarn (for Netlify CLI)
- Netlify CLI (`npm install -g netlify-cli@latest`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/webxos/webxos.git
   cd webxos
   ```

2. **Download Static Assets**:
   ```bash
   curl -o static/dexie.min.js https://cdn.jsdelivr.net/npm/dexie@3.2.4/dist/dexie.min.js
   curl -o static/redaxios.min.js https://cdn.jsdelivr.net/npm/redaxios@0.5.1/dist/redaxios.min.js
   curl -o static/icon.png https://cdn.jsdelivr.net/gh/xai-org/grok-assets/icon.png
   ```

3. **Install Backend Dependencies**:
   ```bash
   pip install -r vial/requirements.txt
   ```

4. **Set Up Environment**:
   Create `vial/.env`:
   ```plaintext
   MCP_HOST=0.0.0.0
   MCP_PORT=5000
   API_TOKEN=secret-token
   WEB3_PROVIDER=https://ropsten.infura.io/v3/YOUR_PROJECT_ID
   ```

## Rebuild Instructions
### Local Development
1. **Start Backend and Database**:
   ```bash
   docker-compose up -d
   ```
   Or run directly:
   ```bash
   cd vial
   uvicorn server:app --host 0.0.0.0 --port 5000
   ```

2. **Frontend**:
   Open `vial.html` in a browser, pointing to `http://localhost:5000`.

3. **Verify**:
   ```bash
   curl http://localhost:5000/health
   ```
   Expected: `{"status": "ok"}`

### Netlify Deployment
1. **Link to Netlify**:
   ```bash
   netlify init
   ```
   Link to `github.com/webxos/webxos`, set publish directory to `.`, and build command to `echo "No build required"`.

2. **Deploy**:
   ```bash
   netlify deploy --prod
   ```
   Update `vial.html` to use the deployed backend URL if hosted separately.

## Usage
- **Authenticate**: Use "Authenticate" in `vial.html` with network and session IDs.
- **Train Vials**: Upload `.md`/`.py` files to train agents.
- **Export**: Download `.md` files with wallet and model data.
- **Git Terminal**: Run git commands via API.
- **Stream**: Monitor real-time updates via SSE at `/stream/{network_id}`.

## Testing
```bash
cd vial
pytest tests/ --verbose --junitxml=test-results.xml
```

## CI/CD
GitHub Actions (`ci.yml`) automates testing and Docker builds. Check logs at [github.com/webxos/webxos](https://github.com/webxos/webxos).

## Troubleshooting
- **Static Assets Missing**: Verify `static/` files are downloaded from CDNs.
- **Train Errors**: Select a file before training.
- **API Errors**: Ensure non-empty prompts in git terminal.
- **Netlify Issues**: Check `netlify.toml` and repo linkage.

## Contributing
Fork, create a feature branch, commit, and submit pull requests to [github.com/webxos/webxos](https://github.com/webxos/webxos).

## License
MIT License. See `LICENSE`.

## Contact
- GitHub: [github.com/webxos/webxos](https://github.com/webxos/webxos)
- X: [x.com/webxos](https://x.com/webxos)
- Email: [contact@webxos.netlify.app](mailto:contact@webxos.netlify.app)
