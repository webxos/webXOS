from typing import Dict
from vial_manager import VialManager
from webxos_wallet import WebXOSWallet
import logging

logger = logging.getLogger(__name__)

class ExportManager:
    def __init__(self, vial_manager: VialManager, wallet: WebXOSWallet):
        self.vial_manager = vial_manager
        self.wallet = wallet

    def export_to_markdown(self, token: str, network_id: str) -> str:
        """Export vial states and wallet data to markdown."""
        try:
            vials = self.vial_manager.get_vials()
            wallet_data = self.wallet.export_wallet(network_id)
            markdown = f"# Vial MCP Export\n\n## Token: {token}\n## Network ID: {network_id}\n\n## Vial States\n"
            for vial_name, state in vials.items():
                markdown += f"### {vial_name}\n{state}\n"
            markdown += "\n## Wallet Data\n" + wallet_data
            logger.info(f"Exported vials for network: {network_id}")
            return markdown
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-11T05:46:00Z]** Export error: {str(e)}\n")
            raise