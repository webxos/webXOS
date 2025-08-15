webxos-backend/
├── src/
│   ├── config/
│   │   └── index.js              # Configuration (env variables, constants)
│   ├── middleware/
│   │   └── auth.js               # Authentication middleware
│   ├── routes/
│   │   ├── wallet.js             # Wallet endpoint
│   │   ├── credentials.js        # Credential generation endpoint
│   │   ├── oauth.js              # OAuth token endpoint
│   │   ├── quantum.js            # Quantum link and troubleshoot endpoints
│   │   ├── api-config.js         # API configuration endpoint
│   │   ├── auth.js               # Authentication endpoint
│   │   ├── void.js               # Void transaction endpoint
│   │   └── vial.js               # Vial data endpoint
│   ├── services/
│   │   └── walletService.js      # Wallet data parsing and validation
│   ├── utils/
│   │   └── logger.js             # Logging utility
│   └── index.js                  # Main server entry point
├── .env                          # Environment variables
├── package.json                  # Project dependencies and scripts
├── README.md                     # Project overview and setup
├── CONTRIBUTING.md               # Open-source contribution guide
└── PROJECT_GUIDE.md              # Detailed project guide for developers
