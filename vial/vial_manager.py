from typing import Dict, List
from agents.agent1 import Agent1
from agents.agent2 import Agent2
from agents.agent3 import Agent3
from agents.agent4 import Agent4
from webxos_wallet import WebXOSWallet
import importlib
import pkgutil
import logging
import re

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
            # Validate .md content
            if not re.search(r'## Vial Data', content) or not re.search(r'wallet', content, re.IGNORECASE):
                logger.error(f"Invalid .md format for {filename}")
                raise ValueError("Invalid .md format: Missing Vial Data or wallet section")
            balance_earned = 0.0
            for vial_name, agent in self.vials.items():
                agent.train(content, filename)
                balance_earned += self.wallet.update_balance(network_id, 0.1)
                logger.info(f"Trained {vial_name} for network {network_id}")
            self.network_state[network_id] = {"last_trained": filename}
            return balance_earned
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- [{new Date().toISOString()}] Training error: {str(e)}\n")
            raise

    def reset_vials(self):
        """Reset all vials and clear network state."""
        try:
            for vial_name, agent in self.vials.items():
                agent.reset()
                logger.info(f"Reset {vial_name}")
            self.network_state.clear()
        except Exception as e:
            logger.error(f"Reset error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- [{new Date().toISOString()}] Reset error: {str(e)}\n")
            raise

    def get_vials(self) -> Dict:
        """Return current state of all vials."""
        try:
            return {name: agent.get_state() for name, agent in self.vials.items()}
        except Exception as e:
            logger.error(f"Get vials error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- [{new Date().toISOString()}] Get vials error: {str(e)}\n")
            raise

    def galaxy_search(self, query: str, vials: List[str]) -> List:
        """Perform agentic web crawl search with specified vials."""
        try:
            # Mock implementation (replace with actual web crawl logic)
            results = [{'item': {'path': '/mock', 'source': 'mock', 'text': {'content': query, 'keywords': [query]}}, 'matches': [{'value': query, 'indices': [[0, len(query)]]}]}]
            logger.info(f"Galaxy search executed for query: {query}, vials: {vials}")
            return results
        except Exception as e:
            logger.error(f"Galaxy search error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- [{new Date().toISOString()}] Galaxy search error: {str(e)}\n")
            raise

    def dna_reasoning(self, query: str, vials: List[str]) -> List:
        """Perform quantum reasoning with specified vials."""
        try:
            # Mock implementation (replace with actual reasoning logic)
            results = [f"Reasoned response for {query} using vials {', '.join(vials)}"]
            logger.info(f"DNA reasoning executed for query: {query}, vials: {vials}")
            return results
        except Exception as e:
            logger.error(f"DNA reasoning error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- [{new Date().toISOString()}] DNA reasoning error: {str(e)}\n")
            raise
