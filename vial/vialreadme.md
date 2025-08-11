Vial MCP Controller
A lightweight, standalone HTML5 Progressive Web App (PWA) for managing and training AI agents in a decentralized, quantum-inspired network. Built with zero external dependencies, it leverages modern JavaScript, Web Crypto API, and a retro terminal aesthetic for seamless operation on edge devices, supporting both online and offline modes.
Table of Contents

Overview
Features
Architecture
Build Structure
Setup and Deployment
Usage
API Commands
Diagrams
Contributing
License

Overview
Vial MCP Controller is a single-page HTML application designed for managing AI agent training within a decentralized WebXOS network. It supports secure wallet management, agent training via a "Quantum Link" mechanism, and seamless import/export of training data and wallets via .md files. The app operates in both online and offline modes, with a focus on lightweight, dependency-free code optimized for mobile and desktop browsers (Chrome, Edge).
Key highlights:

Zero Dependencies: All functionality is contained in a single vial.html file using native JavaScript and Web Crypto API.
Retro Terminal UI: Neon green text on a black background with a Courier New font, inspired by classic terminal interfaces.
Quantum Link: Replaces traditional training with a dynamic agent activation system, supporting online syncing and offline agent generation.
Wallet Management: Combines and shares wallets securely via .md files, with $WEBXOS balance and reputation tracking.
Offline Support: Full training and import functionality in offline mode, with clear UI feedback (grey status bars).

Features

Authentication:

Toggle between online (API and $WEBXOS earning enabled) and offline (local training only) modes.
Generates unique agenticNetworkId and wallet for secure transactions.


Quantum Link:

Activates AI agents for training, setting status to running with yellow UI feedback (#ff0).
Online: Syncs agents with a simulated server, updating quantumState.entanglement to synced.
Offline: Generates local agents from templates or imported data, setting quantumState.entanglement to local.


Auto-Train on Import:

Automatically triggers Quantum Link after importing a .md file, merging tasks, parameters, and wallet data.


Wallet Combining and Sharing:

Imports merge wallet balances and agent data (tasks, training data, configs) seamlessly.
Exports include wallet, blockchain, API credentials, and vial states in a .md file for sharing.


API Training:

Supports /prompt and /task commands (e.g., /prompt vial1 train dataset) to trigger training via LangChain integration (online only).
Offline API attempts display red error messages.


UI Feedback:

Status bars show:
Green (#0f0) for online mode with latency.
Yellow (#ff0) during training.
Grey (#666) for offline mode.
Red (#f00) for errors or offline restrictions.


Console logs commands, errors, and system events with timestamps and IDs.


Security:

Uses Web Crypto API for AES-256 encryption and SHA-256 hashing.
Sanitizes inputs to prevent XSS and validates .md files for safe imports.



Architecture
The Vial MCP Controller is a single-page application with a modular design, leveraging native browser APIs for performance and portability. Key components:

HTML Structure:

Single vial.html file with inline CSS and JavaScript.
Responsive layout with a console, button group, prompt input, vial status bars, and footer.


CSS:

Retro terminal aesthetic: black background, neon green text (#0f0), Courier New font.
Responsive design with media queries for screens as small as 320px.
Dynamic status bar colors: green (online), yellow (training), grey (offline), red (errors).


JavaScript Modules:

State Management: Global state for vials, wallet, blockchain, and API credentials.
Quantum Link: Handles agent activation and training, with online/offline logic.
Import/Export: Merges wallet balances and agent data, exports to .md files.
Git Command Handler: Processes /prompt, /task, /config, /status commands for API-driven training.
Blockchain Simulation: Tracks transactions (auth, train, import, export) with SHA-256 hashes.
Error Handling: Debounced console updates and notification popups for user feedback.


Web Crypto API:

AES-256 for encrypting training data.
SHA-256 for blockchain hashes and wallet verification.



Build Structure
The project consists of a single file, ensuring simplicity and portability:
vial-mcp-controller/
├── vial.html        # Main application file (HTML, CSS, JavaScript)
├── README.md        # This file
└── LICENSE          # MIT License

File Details

vial.html:

Size: ~15KB (unminified).
Structure:
<head>: Meta tags for viewport, charset, and PWA compatibility.
<style>: Inline CSS for retro UI, responsive design, and dynamic status bars.
<body>: Console (<div id="console">), buttons, prompt input, status bars, footer, and hidden file input.
<script>: Inline JavaScript with agent templates, state management, and core functionality.


Dependencies: None (uses native browser APIs: Web Crypto, File API, localStorage).
Version: v2.7 (August 2025).


README.md:

Comprehensive documentation with setup, usage, and diagrams.
Deployable to GitHub Pages or Netlify for instant access.



Setup and Deployment
Prerequisites

Modern browser (Chrome 120+, Edge 120+).
Optional: HTTPS server for online mode (e.g., Netlify, GitHub Pages).
No external dependencies or build tools required.

Local Development

Clone the repository:git clone https://github.com/your-username/vial-mcp-controller.git
cd vial-mcp-controller


Open vial.html in a browser:open vial.html

Or serve locally using a simple HTTP server:python3 -m http.server 8000

Access at http://localhost:8000/vial.html.

Deployment

Netlify:

Drag and drop vial.html into Netlify's deployment interface.
Set custom headers for correct MIME type:/*.html
  Content-Type: text/html


Deploy as a single-file PWA.


GitHub Pages:

Push to a gh-pages branch:git add vial.html
git commit -m "Deploy Vial MCP Controller"
git push origin main


Enable GitHub Pages in repository settings, pointing to gh-pages or main.


Custom Headers:

Ensure Content-Type: text/html for vial.html.
Optional: Add Content Security Policy (CSP) for enhanced security:<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self';">





Usage

Open the App:

Load vial.html in a supported browser.
The UI displays a console, buttons (Authenticate, Void, Troubleshoot, Quantum Link, Export, Import, API Access), prompt input, status bars, and footer.


Authenticate:

Click Authenticate to choose online or offline mode.
Online: Enables API access and $WEBXOS earning.
Offline: Enables local training, with grey status bars (#666).


Quantum Link:

Click Quantum Link to activate agents:
Online: Syncs agents, sets yellow status bars (#ff0).
Offline: Generates local agents, keeps grey status bars.


First-time users: Initializes new wallet and agents automatically.


Import/Export:

Import: Click Import, select a .md file, and auto-train agents with merged wallet and data.
Export: Click Export to download a .md file with wallet, blockchain, API credentials, and vial data for sharing.


API Training:

In online mode, enter commands in the prompt input (e.g., /prompt vial1 train dataset).
Click API Access to generate/view credentials for LangChain integration.


Troubleshoot:

Click Troubleshoot to log system stats and blockchain integrity.


Void:

Click Void to reset all data (vials, wallet, blockchain).



Example Workflow

Authenticate in online mode.
Run /prompt vial1 train dataset to train vial1.
Export data to vial_wallet_export_*.md.
Share the .md file with another user.
They import it, combining wallets, and click Quantum Link to continue training.

API Commands
The app supports git-style commands for API-driven training (online mode only):



Command
Description
Example



/help
Lists available commands
/help


/prompt <vial> <text>
Sends a training prompt to a vial
/prompt vial1 train dataset


/task <vial> <task>
Assigns a task to a vial
/task vial2 optimize_model


/config <vial> <key> <value>
Sets a vial configuration
/config vial3 lr 0.01


/status
Shows vial statuses
/status


Note: Commands containing "train" or "optimize" trigger Quantum Link automatically.
Diagrams
System Architecture
graph TD
    A[Browser] -->|Loads| B[vial.html]
    B --> C[HTML Structure]
    B --> D[CSS (Retro UI)]
    B --> E[JavaScript Modules]
    C --> F[Console]
    C --> G[Button Group]
    C --> H[Prompt Input]
    C --> I[Status Bars]
    C --> J[Footer]
    D --> K[Neon Green Theme]
    D --> L[Responsive Design]
    E --> M[State Management]
    E --> N[Quantum Link]
    E --> O[Import/Export]
    E --> P[Git Command Handler]
    E --> Q[Blockchain Simulation]
    E --> R[Web Crypto API]
    M --> S[Vials]
    M --> T[Wallet]
    M --> U[Blockchain]
    M --> V[API Credentials]
    N -->|Online| W[Agent Sync]
    N -->|Offline| X[Local Agent Generation]
    O --> Y[Markdown .md]
    P -->|API Training| N
    Q --> R
    R -->|AES-256| Z[Encryption]
    R -->|SHA-256| AA[Hashing]

Workflow
sequenceDiagram
    actor User
    participant Browser
    participant VialMCP
    participant Blockchain
    User->>Browser: Open vial.html
    Browser->>VialMCP: Load HTML, CSS, JS
    User->>VialMCP: Click Authenticate
    VialMCP->>Blockchain: Add auth block
    VialMCP->>User: Choose online/offline
    User->>VialMCP: Click Quantum Link
    alt Online Mode
        VialMCP->>Blockchain: Add train block
        VialMCP->>VialMCP: Sync agents (yellow UI)
    else Offline Mode
        VialMCP->>VialMCP: Generate local agents (grey UI)
    end
    User->>VialMCP: Import .md file
    VialMCP->>Blockchain: Add import block
    VialMCP->>VialMCP: Merge wallet, train agents
    User->>VialMCP: Enter /prompt vial1 train dataset
    alt Online Mode
        VialMCP->>Blockchain: Add command block
        VialMCP->>VialMCP: Train vial1 (yellow UI)
    else Offline Mode
        VialMCP->>User: Show error (red UI)
    end
    User->>VialMCP: Click Export
    VialMCP->>Blockchain: Add export block
    VialMCP->>User: Download .md file

UI Layout
graph TD
    A[Body] --> B[H1: Vial MCP Controller]
    A --> C[Console (#console)]
    A --> D[Error Notification (#error-notification)]
    A --> E[API Popup (#api-popup)]
    A --> F[Button Group (.button-group)]
    A --> G[Prompt Input (#prompt-input)]
    A --> H[Vial Status Bars (#vial-status-bars)]
    A --> I[Footer]
    A --> J[File Input (#file-input, hidden)]
    F --> K[Authenticate]
    F --> L[Void]
    F --> M[Troubleshoot]
    F --> N[Quantum Link]
    F --> O[Export]
    F --> P[Import]
    F --> Q[API Access]
    H --> R[Vial 1]
    H --> S[Vial 2]
    H --> T[Vial 3]
    H --> U[Vial 4]

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request with detailed descriptions.

Please ensure:

Code remains dependency-free and lightweight.
Maintains retro terminal aesthetic (neon green, black background, Courier New).
Tests pass on Chrome and Edge (120+).
No external libraries beyond native browser APIs.

License
MIT License. See LICENSE for details.
