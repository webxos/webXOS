import uuid
from typing import Dict, List
from quantum_simulator import QuantumSimulator
from webxos_wallet import WebXOSWallet
from agents.agent1 import run_agent as run_agent1
from agents.agent2 import run_agent as run_agent2
from agents.agent3 import run_agent as run_agent3
from agents.agent4 import run_agent as run_agent4
import logging

logger = logging.getLogger(__name__)

class VialManager:
    def __init__(self, wallet: WebXOSWallet):
        self.wallet = wallet
        self.quantum_sim = QuantumSimulator()
        self.vials: Dict[str, Dict] = {
            f"vial{i+1}": {
                "id": f"vial{i+1}",
                "status": "stopped",
                "code": "",
                "code_length": 0,
                "is_python": True,
                "webxos_hash": str(uuid.uuid4()),
                "wallet": {"address": wallet.create_address(), "balance": 0.0},
                "tasks": []
            } for i in range(4)
        }
        self.agent_runners = {
            "vial1": run_agent1,
            "vial2": run_agent2,
            "vial3": run_agent3,
            "vial4": run_agent4
        }

    def train_vials(self, network_id: str, code: str, filename: str) -> float:
        try:
            balance_earned = 0.0004
            for vial_id, vial in self.vials.items():
                vial["code"] = code
                vial["code_length"] = len(code)
                vial["is_python"] = filename.endswith('.py')
                vial["status"] = "running"
                vial["tasks"] = self.quantum_sim.simulate_training(vial_id, code)["tasks"]
                vial["wallet"]["balance"] += 0.0001
                self.wallet.add_balance(vial["wallet"]["address"], 0.0001)
                self.agent_runners[vial_id](code)  # Run the corresponding agent
            logger.info(f"Trained vials for network: {network_id}")
            return balance_earned
        except Exception as e:
            logger.error(f"Train vials error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-10T20:23:00Z]** Train vials error: {str(e)}\n")
            raise

    def reset_vials(self):
        try:
            for vial in self.vials.values():
                vial["status"] = "stopped"
                vial["code"] = ""
                vial["code_length"] = 0
                vial["is_python"] = True
                vial["webxos_hash"] = str(uuid.uuid4())
                vial["tasks"] = []
                vial["wallet"]["balance"] = 0.0
            logger.info("Vials reset")
        except Exception as e:
            logger.error(f"Reset vials error: {str(e)}")
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-10T20:23:00Z]** Reset vials error: {str(e)}\n")
            raise

    def get_vials(self) -> Dict:
        return self.vials
