# Vial MCP Wallet (BETA) - User Guide

## Introduction
The **Vial MCP Wallet** is a decentralized application (DApp) designed for managing and training AI agents within the WebXOS ecosystem. It provides a blockchain-based interface for interacting with four AI agents (referred to as "vials") and managing a cryptocurrency called `$WEBXOS`. The wallet operates in both online and offline modes, with features for authentication, agent training, blockchain transactions, and API integration. This guide explains how the system works, including agent validation, the `$WEBXOS` conceptual blockchain, and key functionalities.

---

## System Overview
The Vial MCP Wallet is a browser-based application built with HTML, CSS, and JavaScript, designed to simulate a decentralized AI training and wallet management system. It supports the following core features:
- **Agent Management**: Four AI agents (vials), each with simulated PyTorch-based neural network code, can be trained, configured, and tasked.
- **$WEBXOS Token**: A conceptual cryptocurrency used for rewarding agent activity and tracking reputation.
- **Blockchain Simulation**: A local blockchain records transactions, agent states, and system events.
- **Online/Offline Modes**: Supports offline training with limited functionality and online mode for full blockchain and API interactions.
- **API Integration**: Allows interaction with external systems (e.g., LangChain) using generated credentials.
- **Quantum Link**: A feature for synchronizing or generating agent states, simulating advanced AI training.

---

## How the System Validates Agents
The Vial MCP Wallet ensures the integrity and functionality of its agents (vials) through a robust validation process:

1. **Initialization**:
   - Each vial is initialized with a predefined PyTorch-based neural network template (simulated in JavaScript as `agentTemplates`).
   - Vials are assigned unique identifiers (`vial1` to `vial4`), a wallet address, and a `$WEBXOS` hash for tracking.

2. **Code Validation**:
   - Each vial’s code is stored as a string and checked for length and language (Python by default).
   - During import, code blocks in Markdown files are validated to ensure they are Python-based and free of malicious content (e.g., `<script>` tags).

3. **Task and Configuration Validation**:
   - Tasks assigned via `/task` commands are sanitized to prevent injection attacks and stored uniquely to avoid duplication.
   - Configuration parameters set via `/config` are validated for correct syntax and applied to the respective vial.

4. **Training Data Validation**:
   - Training data imported from Markdown files is parsed for tasks and parameters under specific headers (e.g., `## Tasks`, `- Parameter:`).
   - The system checks for valid Markdown format and sanitizes input to remove dangerous characters or scripts.

5. **Quantum State Management**:
   - Each vial has a `quantumState` object tracking its entanglement status (`initialized`, `local`, or `synced`).
   - During the "Quantum Link" process, vials are validated to ensure they are in a running state and have valid training data or tasks.

6. **Blockchain Integration**:
   - Agent actions (e.g., training, task assignment) are recorded as blockchain transactions with SHA-256 hashes.
   - The blockchain ensures that agent states are tamper-proof by linking each block to the previous one via a hash.

---

## $WEBXOS Conceptual Blockchain
The `$WEBXOS` token operates within a simulated blockchain that tracks wallet balances, agent activities, and system events. Here’s how it works:

1. **Blockchain Structure**:
   - The blockchain is a local JavaScript array (`blockchain`) storing blocks with the following properties:
     - `type`: The action type (e.g., `auth`, `command`, `token_earn`, `train`, `export`, `import`).
     - `data`: The payload (e.g., wallet address, tasks, or encrypted data).
     - `timestamp`: The time of the transaction.
     - `prevHash`: The hash of the previous block.
     - `hash`: A SHA-256 hash of the block’s data, timestamp, and previous hash.

2. **Token Earning**:
   - In online mode, the wallet earns 1 `$WEBXOS` every 10 seconds (simulated via `setInterval`).
   - Tokens are distributed equally among the four vials’ wallets.
   - Each token earn event is recorded as a blockchain transaction with a unique hash.

3. **Reputation System**:
   - Reputation points are earned alongside `$WEBXOS` tokens and tied to the wallet’s address.
   - Reputation is reset in offline mode or when the system is voided.

4. **Security**:
   - Blockchain transactions use SHA-256 hashing to ensure integrity.
   - Data encryption (AES-256) is applied to sensitive data during export or training.
   - Input sanitization prevents injection attacks in commands and imported files.

5. **Offline Mode**:
   - In offline mode, blockchain transactions are limited to training, import, and export actions.
   - Token earning and API access are disabled to simulate a disconnected environment.

---

## Key Features and Usage

### 1. Authentication
- **Purpose**: Initializes the wallet and agents in online or offline mode.
- **How to Use**:
  - Click the **Authenticate** button and choose online or offline mode.
  - Online mode generates a wallet address, network ID, and enables token earning.
  - Offline mode disables token earning and API access but allows local training.
- **Validation**: Generates a unique `agenticNetworkId` and wallet address, recorded as a blockchain transaction.

### 2. Void
- **Purpose**: Resets the system to its initial state, clearing all data.
- **How to Use**: Click the **Void** button to reset wallet, vials, blockchain, and authentication status.

### 3. Troubleshoot
- **Purpose**: Displays diagnostic information about the system’s state.
- **How to Use**: Click the **Troubleshoot** button to log the current mode and blockchain length.

### 4. Quantum Link
- **Purpose**: Trains or synchronizes agents, simulating advanced AI processing.
- **How to Use**:
  - Click the **Quantum Link** button to activate training.
  - In offline mode, agents are generated locally; in online mode, they are synced with the network.
  - Supports Markdown file input for training data (via file import).
- **Validation**: Ensures authentication and validates input data before processing.

### 5. Export
- **Purpose**: Exports the wallet and vial states as a Markdown file.
- **How to Use**: Click the **Export** button to download a `.md` file containing:
  - Network ID, wallet details, API credentials, blockchain info, and vial states.
  - Instructions for reuse, extension, and sharing.
- **Validation**: Serializes data safely and logs the export as a blockchain transaction.

### 6. Import
- **Purpose**: Imports a previously exported `.md` file to resume training or merge states.
- **How to Use**:
  - Click the **Import** button and select a `.md` file.
  - The system validates the file format, sanitizes input, and merges data with existing vials.
- **Validation**: Ensures the file is a valid Markdown with four vials and no malicious content.

### 7. API Access
- **Purpose**: Generates credentials for external integration (e.g., LangChain).
- **How to Use**:
  - Click the **API Access** button (available in online mode only).
  - Click **Generate New Credentials** to create an API key and secret.
  - Use these credentials with commands like `/prompt`, `/task`, `/config`, or `/status`.
- **Validation**: Ensures online mode and authentication before generating credentials.

### 8. Console Commands
- **Purpose**: Allows interaction with vials via git-like commands.
- **How to Use**:
  - Enter commands in the input field (e.g., `/prompt vial1 train dataset`).
  - Supported commands:
    - `/help`: Lists available commands.
    - `/prompt <vial> <text>`: Sends a prompt to a vial.
    - `/task <vial> <task>`: Assigns a task to a vial.
    - `/config <vial> <key> <value>`: Sets a configuration parameter.
    - `/status`: Displays vial statuses.
- **Validation**: Sanitizes input and checks for restricted keywords (e.g., `system`, `admin`).

---

## User Interface
The interface is designed to resemble a retro terminal with a cyberpunk aesthetic:
- **Console**: Displays system logs, errors, and balance/reputation updates.
- **Vial Status Bars**: Show the status, latency, and mode of each vial.
- **Buttons**: Provide quick access to core functions (Authenticate, Void, etc.).
- **Prompt Input**: Accepts git-like commands for interacting with vials.
- **API Popup**: Displays and generates API credentials.
- **Error Notifications**: Show temporary alerts for errors or warnings.

---

## Security Considerations
- **Input Sanitization**: All user inputs and imported files are sanitized to prevent injection attacks.
- **Encryption**: Training data is encrypted using AES-256 during export or quantum link.
- **Blockchain Integrity**: Each block is hashed with SHA-256 and linked to the previous block.
- **Restricted Keywords**: Commands containing `system`, `admin`, `root`, `eval`, or `exec` are blocked.
- **File Validation**: Imported `.md` files are checked for size (<1MB), format, and content safety.

---

## Getting Started
1. **Open the Application**:
   - Load the HTML file in a modern browser.
2. **Authenticate**:
   - Click **Authenticate** and choose online or offline mode.
3. **Train Agents**:
   - Use **Quantum Link** to train vials or import a `.md` file with training data.
   - Enter commands like `/prompt vial1 train dataset` to interact with vials.
4. **Manage Tokens**:
   - In online mode, earn `$WEBXOS` tokens automatically.
   - Monitor balance and reputation in the console.
5. **Export/Import**:
   - Export your wallet and vials for backup or sharing.
   - Import a `.md` file to resume training.
6. **Use API**:
   - Generate API credentials in online mode for external integration.

---

## Limitations
- **Beta Status**: The system is in development and may have bugs or incomplete features.
- **Simulated Blockchain**: The blockchain is local and not connected to a real network.
- **Offline Mode**: Token earning and API access are disabled in offline mode.
- **File Size Limit**: Imported files must be under 1MB.
- **Browser Dependency**: Requires a modern browser with Web Crypto API support.

---

## Resources
- **Website**: [webxos.netlify.app](https://webxos.netlify.app)
- **GitHub**: [github.com/webxos/webxos](https://github.com/webxos/webxos)
- **License**: MIT Open Source License
- **Disclaimer**: `$WEBXOS` is in development. WebXOS claims no responsibility for token loss.

---

## Example Workflow
1. Authenticate in online mode.
2. Run `/prompt vial1 train dataset` to train `vial1`.
3. Assign a task: `/task vial2 optimize_model`.
4. Set a config: `/config vial3 lr 0.01`.
5. Check status: `/status`.
6. Export the wallet and vials.
7. Import a `.md` file to resume training.
8. Generate API credentials for LangChain integration.

---

This guide provides a comprehensive overview of the Vial MCP Wallet’s functionality, agent validation, and blockchain system. For further assistance, refer to the `/help` command or the resources above.