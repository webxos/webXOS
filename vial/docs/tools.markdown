# Vial MCP Tools

This document describes the tools available in the Vial MCP Controller for interaction with the agentic network and $WEBXOS wallet.

## Available Tools

### sample_tool
- **Description**: Processes input data and returns a formatted response.
- **Input**: String data.
- **Output**: Formatted string response.
- **Example**:
  ```json
  {
    "input": "test data",
    "output": "Processed: test data"
  }
  ```

### train_vials
- **Description**: Trains the four vial agents with uploaded content and updates wallet balance.
- **Input**: File content (text), network ID, and filename.
- **Output**: Vial states and updated balance.
- **Example**:
  ```json
  {
    "networkId": "test_network",
    "file": "test.txt",
    "vials": {"vial1": {"filename": "test.txt", "content_length": 12}, ...},
    "balance": 0.4
  }
  ```

## Usage
Tools are dynamically loaded from the `tools/` directory and exposed via the FastMCP server. Use the `/comms_hub` endpoint or `client.py` to interact with tools.