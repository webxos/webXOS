# Under Development: Agent Grounding: Agent-OEM v1.0 
```
▄▖      ▗   ▄▖▄▖▖  ▖
▌▌▛▌█▌▛▌▜▘▄▖▌▌▙▖▛▖▞▌
▛▌▙▌▙▖▌▌▐▖  ▙▌▙▖▌▝ ▌ v1.0
  ▄▌
```

A drop‑in, modular AI agent backend that serves the Agent Grounding frontend and provides a unified API for 10 agent‑to‑agent protocol phases. The backend is built with a hot‑swappable plugin system (Python) so you can toggle the five core use cases (customer support, repo maintenance, document analysis, fintech auditing, and omni‑onboarding) via a single `config.yaml`.

The index.html internal handlers have calls to the backend. The backend implements 10 phases and integrates a modular plugin system. You can expand each plugin with actual business logic (e.g., Stripe, GitHub, ChromaDB) by adding dependencies and API calls.


## Features

- **10‑Phase Protocol** – Liveness, memory, encryption, task queue, guardrails, payments, negotiation, and more.
- **Modular Plugins** – Each use case is a self‑contained Python class; enable/disable via config.
- **Agent‑First Frontend** – The provided `index.html` (Agent Grounding v2.3.1) gives you a full UI to test every phase.
- **No‑Cloud Privacy** – All state can be kept locally (in‑memory or persistent storage like Redis).
- **Deploy Anywhere** – Docker, serverless (Modal, Vercel), or traditional VM.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agent-oem.git
   cd agent-oem

### Tree Structure
```markdown 
agent-oem/
├── index.html                 # Modified Agent Grounding UI (calls backend API)
├── README.md                  # Full documentation & deployment guide
├── LICENSE                    # MIT License
├── .gitignore                 # Standard Python/git ignores
├── docker-compose.yml         # Easy local dev with Docker
├── Dockerfile                 # Multi‑stage build for backend
├── requirements.txt           # Python dependencies
├── config.yaml                # OEM master switchboard (toggle plugins, LLM keys)
├── main.py                    # FastAPI application – serves UI + /api routes
├── core/
│   ├── __init__.py
│   ├── base_agent.py          # Standard agent interface
│   └── orchestra.py           # Orchestrator that loads plugins from config
└── plugins/
    ├── __init__.py
    ├── customer_support.py    # Use Case 1
    ├── repo_maintainer.py     # Use Case 2
    ├── doc_analyst.py         # Use Case 3
    ├── fintech_auditor.py     # Use Case 4
    └── omni_onboarder.py      # Use Case 5
```
### Overview

Agent-OEM is a modular Python/FastAPI backend providing a unified API for a 10-phase agent-to-agent protocol, with hot-swappable plugins for use cases like customer support, repo maintenance, document analysis, fintech auditing, and omni-onboarding. The system emphasizes no-cloud privacy via local or Redis storage, configurable plugins through config.yaml, and includes a bundled Agent Grounding frontend for testing all phases, deployable via Docker or serverless.


  ### Plugin Development

    Create a new file plugins/my_plugin.py.

    Implement a class that inherits from core.base_agent.BaseAgent.

    Define async def execute(self, action: str, params: dict) -> dict.

    Provide a function initialize_agent() that returns an instance.

    Add your plugin name to enabled_modules in config.yaml.

### Deployment

    GitHub Pages: For static hosting of the UI only (no backend). But to use the full backend, deploy the Docker container on any cloud.

    Modal / Vercel: The backend is a standard FastAPI app – deploy it as a serverless function.

    Render / Fly.io: Use the Dockerfile for one‑click deployment.

### Configuration

Edit config.yaml:

    enabled_modules: set each use case to true or false.

    system.orchestrator_llm: choose the default LLM for orchestration.

    integrations: provide environment variable names for API keys.

The backend automatically loads only the enabled plugins.


### Use Cases

- **Customer Support Chat**: Integrate Agent-OEM's customer_support plugin as the backend for your website's live chat. Users click a "Talk to AI Assistant" button that triggers the 10-phase protocol for natural conversation, ticket creation, and escalation, with plugins handling order lookups or refunds autonomously.

- **Content & Repo Maintenance**: For developer or documentation-heavy sites, add an "AI Maintenance" link/button in the admin dashboard. It activates the repo_maintainer plugin to scan GitHub repos linked to the site, suggest updates, fix docs, or analyze user feedback from site comments.

- **Document Analysis Tool**: On knowledge-base or SaaS websites, provide a "Upload & Analyze" button for PDFs/contracts. The doc_analyst plugin processes files server-side via the backend, returning summaries, insights, or compliance checks while keeping data local for privacy.

- **Fintech/Payments Onboarding**: E-commerce or fintech sites can embed an "AI Financial Advisor" or "Secure Onboard" button. Agent-OEM's fintech_auditor and omni_onboarder plugins handle guided KYC flows, payment negotiation, guardrails, and encryption across the full agent protocol.

- **Personalized Omni-Experience**: Add a prominent floating "AI Companion" button for any site. It loads the full modular backend, letting users switch between plugins (e.g., support + doc analysis) in one session, with memory and task queue for ongoing personalized recommendations or multi-step website automation.

### License

MIT


