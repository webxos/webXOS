import sqlite3
import uuid
from webxos_wallet import WebXOSWallet
from pydantic import BaseModel

class MdSchema(BaseModel):
    token_tag: str
    wallet: dict
    vial_states: dict

class ExportManager:
    def export_to_md(self, network_id, vials):
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('SELECT address, balance FROM wallets WHERE network_id = ?', (network_id,))
        wallet_data = cursor.fetchone()
        conn.close()

        token_tag = f"## WEBXOS Tokenization Tag: {str(uuid.uuid4())}\n"

        wallet_md = f"""## $WEBXOS Wallet
- Address: {wallet_data[0] if wallet_data else 'N/A'}
- Balance: {wallet_data[1] if wallet_data else 0.0}
"""

        vials_md = "## Vial States\n"
        for vial_id, data in vials.items():
            vials_md += f"### {vial_id}\n- Output: {data['output']}\n- Quantum State: {data['quantum_state']}\n"

        md_content = f"{token_tag}{wallet_md}\n{vials_md}"
        # Validate with schema
        schema_data = {
            "token_tag": token_tag.strip(),
            "wallet": {"address": wallet_data[0] if wallet_data else 'N/A', "balance": wallet_data[1] if wallet_data else 0.0},
            "vial_states": vials
        }
        MdSchema(**schema_data)
        return md_content

# [xaiartifact: v1.7]
