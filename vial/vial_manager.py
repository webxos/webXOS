from typing import Dict, List
from agents.agent1 import Agent1
from agents.agent2 import Agent2
from agents.agent3 import Agent3
from agents.agent4 import Agent4
from webxos_wallet import WebXOSWallet
import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)

class VialManager:
    def __init__(self, wallet: WebXOSWallet):
        self.vials = {
            "vial1": Agent1(),
            "vial2": Agent2(),
            "vial3": Agent3(),
            "vial4": Agent4()
        }
        self.tools = self.load_tools()
        self.wallet = wallet
        self.network_state = {}

    def load_tools(self) -> Dict:
        """Dynamically load MCP tools from tools/ directory."""
        tools = {}
        for _, name, _ in pkgutil.iter_modules(['tools']):
            module = importlib.import_module(f'tools.{name}')
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '_is_mcp_tool'):
                    tools[name] = attr
        return tools

    def train_vials(self, network_id: str, content: str, filename: str) -> float:
        """Train all vials with provided content and update wallet balance."""
        try:
            balance_earned = 0.0
            for vial_name, agent in self.vials.items():
                agent.train(content, filename)
                balance_earned += self.wallet.update_balance(network_id, 0.1)
                logger.info(f"Trained {vial_name} for network {network_id}")
            self.network_state[network_id] = {"last_trained": filename}
            return balance_earned
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

    def reset_vials(self):
        """Reset all vials and clear network state."""
        for vial_name, agent in self.vials.items():
            agent.reset()
            logger.info(f"Reset {vial_name}")
        self.network_state.clear()

    def get_vials(self) -> Dict:
        """Return current state of all vials."""
        return {name: agent.get_state() for name, agent in self.vials.items()}