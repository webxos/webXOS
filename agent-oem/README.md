# Under Development

```
▄▖      ▗   ▄▖▄▖▖  ▖
▌▌▛▌█▌▛▌▜▘▄▖▌▌▙▖▛▖▞▌
▛▌▙▌▙▖▌▌▐▖  ▙▌▙▖▌▝ ▌ v1.0
  ▄▌
```

Agent-OEM is a backend scaffolding tool for developing local-first, privacy-focused autonomous agents. It functions as an OEM developer template for embedding or white-labeling agent workflows that operate entirely on local infrastructure. A drop‑in, modular agent backend that serves the frontend and provides API for 10 agent‑to‑agent protocol phases. The backend is built with a hot‑swappable plugin system (Python) so you can toggle the five core use cases (customer support, repo maintenance, document analysis, fintech auditing, and omni‑onboarding) via a single `config.yaml`. The index.html internal handlers have calls to the backend. The backend implements 10 phases and integrates a modular plugin system. You can expand each plugin with actual business logic (e.g., Stripe, GitHub, ChromaDB) by adding dependencies and API calls.


## Features

- **10‑Phase Protocol** – Liveness, memory, encryption, task queue, guardrails, payments, negotiation, and more.
- **Modular Plugins** – Each use case is a self‑contained Python class; enable/disable via config.
- **Agent‑First Frontend** – The provided `index.html` (Agent Grounding v2.3.1) gives you a full UI to test every phase.
- **No‑Cloud Privacy** – All state can be kept locally (in‑memory or persistent storage like Redis).
- **Deploy Anywhere** – Docker, serverless (Modal, Vercel), or traditional VM.

### Tree Structure
```markdown 
agent-oem/
├── index.html                
├── README.md                  
├── LICENSE                    
├── .gitignore                 
├── docker-compose.yml        
├── Dockerfile            
├── requirements.txt          
├── config.yaml               
├── main.py                    
├── core/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── crypto.py            
│   └── orchestra.py           
└── plugins/
    ├── __init__.py
    ├── customer_support.py
    ├── repo_maintainer.py
    ├── doc_analyst.py
    ├── fintech_auditor.py
    └── omni_onboarder.py
```
### Overview

Agent-OEM is a modular Python/FastAPI backend providing a unified API for a 10-phase agent-to-agent protocol, with hot-swappable plugins for use cases like customer support, repo maintenance, document analysis, fintech auditing, and omni-onboarding. The system emphasizes no-cloud privacy via local or Redis storage, configurable plugins through config.yaml, and includes a bundled Agent Grounding frontend for testing all phases, deployable via Docker or serverless.


  ### Plugin Development

  Create a new file plugins/my_plugin.py.

  Implement a class that inherits from core.base_agent.BaseAgent:

    Define async def execute(self, action: str, params: dict) -> dict.

    Provide a function initialize_agent() that returns an instance.

    Add your plugin name to enabled_modules in config.yaml.

### Deployment

  Deploy the Docker container on any cloud:

  **Modal / Vercel:** The backend is a standard FastAPI app – deploy it as a serverless function.

  **Render / Fly.io:** Use the Dockerfile for one‑click deployment.

### Configuration

Edit config.yaml:

    enabled_modules: set each use case to true or false.

    system.orchestrator_llm: choose the default LLM for orchestration.

    integrations: provide environment variable names for API keys.

The backend automatically loads only the enabled plugins.


### Use Cases

- **Autonomous Support Agent**: Embed Agent-OEM backend in your website's API layer so external customer agents or chatbots can call its unified 10-phase endpoint. Other agents negotiate refunds, check orders, or escalate issues via protocol phases like memory, guardrails, and payments without exposing any frontend.

- **Content Sync Agent**: Integrate the repo_maintainer and doc_analyst plugins server-side. Your website's internal agent or third-party automation agents query the backend API to analyze site content, update docs, or maintain linked repositories autonomously through task queue and negotiation phases.

- **Fintech Transaction Agent**: For payment-heavy websites, expose secure Agent-OEM endpoints so partner financial agents can initiate audited transactions, KYC flows, or negotiations. The backend handles encryption, guardrails, and payments plugin logic purely agent-to-agent.

- **Omni-Onboarding Agent**: Embed the omni_onboarder plugin for website user flows where external identity or CRM agents call the backend to complete multi-step onboarding. It manages liveness checks, memory persistence, and negotiation across distributed agent systems.

- **Multi-Agent Orchestration Hub**: Use the core orchestra and all plugins as a central backend service. Your website acts as host for swarms of specialized agents (internal or from partners) that interact via the 10-phase protocol for complex tasks like coordinated support + analysis + maintenance sessions.

### License

MIT


