vial-mcp-project/
├── .env                             # Environment variables
├── Dockerfile                       # Docker configuration
├── LICENSE.txt                      # Project license
├── ci.yml                           # GitHub Actions CI/CD configuration
├── docker-compose.yaml              # Docker Compose for multi-service deployment
├── netlify.toml                     # Netlify deployment configuration
├── vial.html                        # Main frontend (retro terminal UI)
├── chatbot.html                     # Primary chatbot UI
├── README.md                        # Project overview and setup
├── errorlog.md                      # Conversation error log
├── opensourceguide.md               # Open-source contribution guide
├── backup_restore.py                # Backup and restore script
├── /chatbot/                        # Chatbot-related files
│   ├── chatbot2.html                # Secondary chatbot UI
│   ├── nano_gpt_bots.md            # Chatbot documentation
│   ├── server.py                   # Chatbot backend
│   ├── site_index.json             # Chatbot and site search index
│   └── sw.js                       # Chatbot service worker
├── /db/                             # Database-related files
│   ├── dbreadme.md                  # Database documentation
│   ├── dockerfile                   # Database Docker configuration
│   ├── mcp_db_init.py              # MongoDB initialization
│   ├── nlp_model.py                # NLP model for query enhancement
│   ├── requirements.txt             # Database dependencies
│   ├── server.py                   # Database server
│   ├── wait-for-it.sh              # Utility script for service dependencies
│   ├── errorlog.md                 # Database error logs
│   ├── library_agent.py             # Library agent for vials
│   ├── nomic_sw.js                  # Service worker for Nomic vial
│   ├── llmware_sw.js                # Service worker for LLMware vial
│   ├── jina_sw.js                   # Service worker for Jina AI vial
│   ├── global_mcp_agents.py         # Global MCP agents for library sync
│   ├── translator_agent.py          # Language translator agent
│   ├── library_sync.py              # Library synchronization
│   ├── test_library_agent.py        # Unit tests for library_agent
│   ├── test_translator_agent.py     # Unit tests for translator_agent
│   ├── cognitallmware_sw.js         # Service worker for CogniTALLMware vial
│   ├── library_config.json          # Library configuration settings
│   ├── monitor_agent.py             # Library performance monitoring
│   ├── test_library_sync.py         # Unit tests for library_sync
│   ├── test_monitor_agent.py        # Unit tests for monitor_agent
│   ├── auth_sync.py                 # Authentication synchronization
│   ├── cache_manager.py             # Caching for library responses
│   ├── audit_log.py                 # Audit logging
│   ├── test_auth_sync.py            # Unit tests for auth_sync
│   ├── test_cache_manager.py        # Unit tests for cache_manager
│   ├── performance_metrics.py       # Performance metrics collection
│   ├── config_validator.py          # Configuration validation
│   ├── health_check.py              # Health check endpoints
│   ├── test_performance_metrics.py  # Unit tests for performance_metrics
│   ├── test_config_validator.py     # Unit tests for config_validator
├── /vial/                           # Core Vial MCP backend and logic
│   ├── __init__.py                 # Package initializer
│   ├── auth_manager.py             # JWT authentication
│   ├── client.py                   # MCP client
│   ├── export_manager.py           # Markdown export functionality
│   ├── langchain_agent.py          # LangChain integration
│   ├── network_sync.py             # Network synchronization agent
│   ├── quantum_simulator.py        # Quantum simulation with Qiskit
│   ├── unified_server.py           # Unified FastAPI backend
│   ├── vial_manager.py             # Vial agent management
│   ├── vial_server.py              # Vial-specific server
│   ├── webxos_wallet.py            # WebXOS wallet management
│   ├── database.sqlite             # SQLite database for wallet
│   ├── errorlog.md                 # Vial error logs
│   ├── vial_wallet_export_*.md     # Wallet export files
│   ├── /agents/                    # Agent scripts
│   │   ├── agent1.py               # Agent 1 script (Nomic)
│   │   ├── agent2.py               # Agent 2 script (CogniTALLMware)
│   │   ├── agent3.py               # Agent 3 script (LLMware)
│   │   └── agent4.py               # Agent 4 script (Jina AI)
│   ├── /docs/                      # Vial documentation
│   │   ├── api.markdown            # API documentation
│   │   ├── setup.markdown          # Setup instructions
│   │   ├── openapi.yaml           # OpenAPI specification
│   │   ├── resources.md            # Resource documentation
│   │   ├── tools.md               # Tools documentation
│   │   └── vial_readme.md         # Vial-specific README
│   ├── /prompts/                   # Prompt files
│   │   ├── base_prompt.py         # Base prompt for LangChain
│   │   └── blank.txt              # Placeholder file
│   ├── /static/                    # Vial static assets
│   │   ├── dexie.min.js           # Dexie.js for offline storage
│   │   ├── icon.png               # App icon
│   │   └── redaxios.min.js        # Redaxios for HTTP requests
│   ├── /tests/                     # Test suite
│   │   ├── test_auth_manager.py   # Auth tests
│   │   ├── test_export_manager.py # Export tests
│   │   ├── test_langchain_agent.py # LangChain tests
│   │   ├── test_quantum_simulator.py # Quantum simulator tests
│   │   ├── test_server.py         # Server tests
│   │   ├── test_vial_manager.py   # Vial manager tests
│   │   ├── test_wallet.py         # Wallet tests
│   │   └── test_vial_html.js      # Frontend tests
│   └── /tools/                     # Extensible tools
│       └── sample_tool.py         # Sample tool
├── /node/                           # Node.js authentication server
│   ├── server.js                   # Node.js server
│   ├── package.json                # Node.js dependencies
│   ├── /src/                       # Node.js source files
│   │   ├── network_sync.js        # NetworkSyncAgent
│   │   └── auth_verification.js   # AuthVerificationAgent
├── /static/                         # Root-level static assets
│   ├── agent1.js                   # Agent 1 frontend script
│   ├── agent2.js                   # Agent 2 frontend script
│   ├── agent3.js                   # Agent 3 frontend script
│   ├── agent4.js                   # Agent 4 frontend script
│   ├── fuse.min.js                 # Fuzzy search for agents
│   ├── neurots.js                  # Neural network visualization
│   ├── style.css                   # Global styles
│   └── site_index.json             # Search index for MCP tools
├── /docs/                           # Root-level documentation
│   ├── api.markdown                # Root API documentation
│   ├── setup.markdown              # Root setup instructions
│   ├── openapi.yaml               # Root OpenAPI specification
│   └── vial_readme.md              # Root project README
├── /prompts/                        # Root-level prompts
│   ├── base_prompt.py              # Copied base prompt
│   └── blank.txt                   # Placeholder file
├── /resources/                      # Extensible resources
│   └── mcp_tools.json              # MCP tool configurations
└── /tools/                          # Root-level tools
    ├── sample_tool.py              # Sample tool
    └── webpage_creator.py          # Webpage creation tool
