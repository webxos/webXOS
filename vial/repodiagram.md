webXOS/vial/
├── Dockerfile # Docker setup
├── docker-compose.yml # Auto-starts server and WebSocket
├── mcp.json # MCP config with OAuth
├── schemas/
│ ├── vial.schema.json # Vial schema
│ ├── log.schema.json # Log schema
│ ├── training.schema.json # Training schema
│ ├── oauth.schema.json # OAuth schema
│ └── wallet.schema.json # Wallet schema for $WEBXOS tokens
├── src/
│ ├── server.js # Node.js server with SQLite, OAuth, and wallet endpoints
│ ├── tools/
│ │ ├── vial_manager.js # Vial logic
│ │ ├── log_manager.js # Log handling
│ │ ├── training.js # Training logic
│ │ ├── oauth.js # OAuth handling
│ │ ├── diagnostics.js # System diagnostics
│ │ └── walletagent.js # Wallet and blockchain logic for $WEBXOS
├── static/
│ ├── vial.html # Main UI with embedded logic
│ ├── wallet.html # Wallet UI for $WEBXOS PoW token
│ ├── vial3d.html # 3D Markdown console for vial agents
│ ├── icon.png # Placeholder icon
│ ├── manifest.json # Web app manifest
│ ├── sql-wasm.wasm # SQLite WASM
│ ├── worker.js # Web Worker for SQLite
│ ├── redaxios.min.js # HTTP client (~800 bytes)
│ ├── lz-string.min.js # Compression (~1 kB)
│ ├── mustache.min.js # Templating (~9 kB)
│ ├── dexie.min.js # IndexedDB (~20 kB)
│ ├── jwt-decode.min.js # JWT decoding (~5 kB)
│ ├── three.min.js # Three.js for 3D rendering (~500 kB)
│ ├── webgpu-polyfill.min.js # WebGPU polyfill (~100 kB)
├── uploads/
│ ├── templates/ # Markdown templates
│ └── outputs/ # Exported logs/vials
├── scripts/
│ ├── build.sh # Build script
│ ├── setup.sh # Setup script
│ ├── test.sh # Test script
├── tests/
│ ├── unit/
│ │ ├── vial_manager.test.js
│ │ ├── log_manager.test.js
│ │ ├── training.test.js
│ │ ├── oauth.test.js
│ │ ├── diagnostics.test.js
│ │ └── walletagent.test.js # Tests for walletagent.js
│ └── integration/
│     └── server.test.js
├── .env.example # Environment variables
├── errorlog.md # Error log
├── database.db # SQLite database
├── README.md # Documentation
└── LICENSE # MIT License
