# WebXOS Vial MCP Controller

Remote controller for managing 4x agentic quantum-simulated network vials. Vials are PyTorch-based agents exported as .md files with $WEBXOS wallet data. Users earn $WEBXOS, exportable to a decentralized system for Stripe cash-outs and app/game development.

## Setup
1. Clone repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Build Docker: `docker build -t vial-mcp .`
4. Run: `docker run -p 8080:8080 vial-mcp`
5. Access `vial.html` at `http://localhost:8080/vial.html`

## Features
- **4x Agentic Network**: Auto-initialized vials with PyTorch models.
- **$WEBXOS Wallet**: Earn and export tokens.
- **Offline Fallback**: Continue operations without internet.
- **Authentication**: Secure link with fallback to localStorage.
- **Export/Import**: Save and load vial states as .md files.

## File Structure
- `/vial/`: Backend scripts and Docker setup.
- `vial.html`: Master controller UI.
- `errorlog.md`: Tracks errors with timestamps and analysis.
- `static/`: Static assets (Dexie, Redaxios, icon).

## Notes
- Keep `vial.html` and `errorlog.md` updated.
- Use `xaiartifact` tags for tracking.
- Ensure `/vial/` is rebuildable outside Docker.

## xAI Artifact
- Version: 1.0
- Last Updated: 2025-08-10
