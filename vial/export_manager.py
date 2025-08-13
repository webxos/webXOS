import datetime
import hashlib
import uuid
from vial.webxos_wallet import WebXOSWallet

class ExportManager:
    def __init__(self):
        self.wallet = WebXOSWallet()

    def generate_export(self, user_id: str, vials: dict) -> str:
        try:
            network_id = str(uuid.uuid4())
            wallet_balance = self.wallet.get_balance(user_id)
            wallet_address = str(uuid.uuid4())
            wallet_hash = hashlib.sha256(f"{user_id}{datetime.datetime.utcnow().isoformat()}".encode()).hexdigest()
            export_content = f"""# WebXOS Vial and Wallet Export

## Agentic Network
- Network ID: {network_id}
- Session Start: {datetime.datetime.utcnow().isoformat()}
- Session Duration: 0.00 seconds
- Reputation: {int(wallet_balance * 1000)}

## Wallet
- Wallet Key: {str(uuid.uuid4())}
- Session Balance: {wallet_balance:.4f} $WEBXOS
- Address: {wallet_address}
- Hash: {wallet_hash}

## API Credentials
- Key: none
- Secret: none

## Blockchain
- Blocks: 11914
- Last Hash: {wallet_hash}

## Vials
"""
            for vial_id, vial_data in vials.items():
                export_content += f"""# Vial Agent: {vial_id}
- Status: {vial_data.get('status', 'running')}
- Language: Python
- Code Length: {len(vial_data.get('script', ''))} bytes
- $WEBXOS Hash: {hashlib.sha256(vial_id.encode()).hexdigest()}
- Wallet Balance: {(wallet_balance / 4):.4f} $WEBXOS
- Wallet Address: {str(uuid.uuid4())}
- Wallet Hash: {wallet_hash}
- Tasks: none
- Quantum State: {{
      "qubits": [],
      "entanglement": "synced"
    }}
- Training Data: [
      {{
        "tasks": [],
        "parameters": {{}},
        "hash": "{hashlib.sha256(str(datetime.datetime.utcnow()).encode()).hexdigest()}"
      }}
    ]
- Config: {{}}

```python
{vial_data.get('script', '')}
```
"""
            return export_content
        except Exception as e:
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Export generation error: {str(e)}\n")
            raise

    def validate_export(self, export_data: str) -> bool:
        return self.wallet.validate_export(export_data)
