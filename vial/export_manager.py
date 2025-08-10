import datetime

class ExportManager:
    def export_vials(self, vials, wallet, network_id, session_start):
        content = f"""# WebXOS Vial and Wallet Export

## Agentic Network
- Network ID: {network_id or 'none'}
- Session Start: {session_start or 'none'}
- Session Duration: {((datetime.datetime.now() - datetime.datetime.fromisoformat(session_start)).seconds if session_start else 0)} seconds

## Wallet
- Wallet Key: {wallet.address or 'none'}
- Session Balance: {wallet.balance:.4f} $WEBXOS
- Address: {wallet.address or 'offline'}

## Vials
"""
        for vial in vials:
            content += f"""# Vial Agent: {vial['id']}
- Status: {vial['status']}
- Language: Python
- Code Length: {vial['codeLength']} bytes
- $WEBXOS Hash: {vial['webxosHash']}
- Wallet Balance: {vial['wallet']['balance']:.4f} $WEBXOS
- Wallet Address: {vial['wallet']['address'] or 'none'}
- Tasks: {', '.join(vial['tasks']) or 'none'}

```python
{vial['code']}
```
---
"""
        with open(f'/data/vial_results/vial_wallet_export_{datetime.datetime.now().isoformat().replace(":", "-")}.md', 'w') as f:
            f.write(content)
        return content
