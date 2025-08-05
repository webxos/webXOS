*IN DEVELOPMENT*
Vial MCP Server
A lightweight, modular, and open-source server for creating and managing "vials" (trainable agents) with support for HTML, CSS, JS, and Python via WebXOS integration. Vials can be trained with NLP/LLM/API data or uploaded files, designed to be copied and used instantly from a single GitHub repository.
Setup

Clone the repository:
git clone <repo-url>
cd vial


Install dependencies:
npm install


Build WebAssembly module:
npm run build-wasm


Run the server:
./run-vial.sh



Usage

Access the interface at http://localhost:8080.
Use the UI to create, troubleshoot, monitor, and export vials.
Upload code files (.html, .css, .js, .py) or provide API URLs for training.
Export vials as .md files for editing and deployment.

Features

File Upload: Supports .html, .css, .js, .py files using Multer.
WebXOS Integration: Trains vials via webxos.netlify.app using TensorFlow.js and Pyodide.
Scalability: MongoDB for vial state persistence.
Security: JWT-based authentication for API endpoints.
Export: Generates .md files for vial configurations.
Docker: Containerized for easy deployment.

Directory Structure
vial/
├── Vial.html
├── mcp-vial.js
├── run-vial.sh
├── Dockerfile
├── package.json
├── .gitignore
├── README.md
└── static/
    ├── vial_network.js
    ├── vial_network.wasm
    ├── style.css
    └── site_index.json

Requirements

Node.js 18+
Docker
Emscripten (for WebAssembly compilation)
MongoDB (for scalability)

License
MIT License
