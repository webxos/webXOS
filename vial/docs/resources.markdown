# Vial MCP Resources

This document describes the resources available in the Vial MCP Controller for read-only data access.

## Available Resources

### health
- **Description**: Checks the health status of the MCP server.
- **Endpoint**: `/health`
- **Output**: Server status.
- **Example**:
  ```json
  {
    "status": "ok"
  }
  ```

### export
- **Description**: Exports vial states and wallet data as markdown for a given network ID.
- **Endpoint**: `/export?networkId={network_id}`
- **Output**: Markdown string with vial and wallet data.
- **Example**:
  ```json
  {
    "markdown": "# Vial MCP Export\n\n## Token: test_token\n## Network ID: test_network\n..."
  }
  ```

## Usage
Resources are accessible via the FastAPI endpoints and exposed as MCP resources through FastMCP. Use `client.py` or HTTP requests to access them.