vial-mcp-project/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions for CI/CD
├── vial/
│   ├── server.py                   # FastAPI server for MCP API
│   ├── vial_manager.py             # Manages 4x vial agents and LangChain integration
│   ├── quantum_simulator.py        # Quantum network simulation logic
│   ├── webxos_wallet.py            # $WEBXOS wallet handling
│   ├── auth_manager.py             # Authentication and offline fallback logic
│   ├── export_manager.py           # Export PyTorch models and wallet data to .md
│   ├── langchain_agent.py          # LangChain agent with NanoGPT for comms
│   ├── agents/
│   │   ├── agent1.py              # Agent for vial1 with PyTorch and LangChain
│   │   ├── agent2.py              # Agent for vial2 with PyTorch and LangChain
│   │   ├── agent3.py              # Agent for vial3 with PyTorch and LangChain
│   │   └── agent4.py              # Agent for vial4 with PyTorch and LangChain
│   ├── tests/
│   │   ├── test_server.py         # Tests for server.py
│   │   └── test_vial_manager.py   # Tests for vial_manager.py
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                 # Docker configuration for backend
├── static/
│   ├── icon.png                   # Favicon for vial.html
│   ├── dexie.min.js               # Dexie.js for offline storage
│   └── redaxios.min.js            # Redaxios for HTTP requests
├── README.md                      # Project overview, setup, and architecture diagram
├── errorlog.md                    # Error tracking log
├── vial.html                      # Master key remote controller
└── LICENSE                        # MIT License
