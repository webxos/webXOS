Wallet File Format
Overview
The .md wallet files are used for importing/exporting wallet data in the Vial2 MCP system, ensuring seamless integration with the $WEBXOS network.
Format
# WebXOS Vial and Wallet Export

## Agentic Network
- Network ID: <uuid>
- Session Start: <timestamp>
- Session Duration: <seconds>
- Reputation: <integer>

## Wallet
- Wallet Key: <uuid>
- Session Balance: <float> $WEBXOS
- Address: <uuid>
- Hash: <sha256_hash>

## API Credentials
- Key: <uuid>
- Secret: <hex_string>

## Blockchain
- Blocks: <integer>
- Last Hash: <sha256_hash>

## Vials
# Vial Agent: <vial_id>
- Status: <running|stopped>
- Language: Python
- Code Length: <bytes>
- $WEBXOS Hash: <uuid>
- Wallet Balance: <float> $WEBXOS
- Wallet Address: <uuid>
- Wallet Hash: <sha256_hash>
- Tasks: <list>
- Quantum State: <json>
- Training Data: <json_array>
- Config: <json>

```python
<python_code>


## Usage
- **Import**: Use the "Import" button in vial2.html to load the `.md` file.
- **Export**: Use the "Export" button to generate a `.md` file from current wallet data.
- **Merge**: Combine multiple wallets using the `/mcp/api/wallet_merge` endpoint.
- **Validation**: Ensure `Wallet Hash` matches backend records for security.

## Example
See `vial_wallet_export_2025-08-18T12-15-17-815Z.md` for a sample wallet file.

# xAI Artifact Tags: #vial2 #docs #wallet_format #neon_mcp
