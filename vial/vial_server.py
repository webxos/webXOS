import uuid
import torch
import torch.nn as nn
from typing import List
from vial.models import Vial, Wallet

class VialManager:
    def generate_uuid(self):
        return str(uuid.uuid4())

    def create_vials(self, network_id: str) -> List[Vial]:
        default_code = """import torch
import torch.nn as nn

class VialAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = VialAgent()"""
        return [
            Vial(
                id=f"vial{i+1}",
                status="stopped",
                code=default_code,
                codeLength=len(default_code),
                isPython=True,
                webxosHash=f"{network_id}-{i+1}",
                wallet=Wallet(address=f"0x{self.generate_uuid()[:10]}", balance=0)
            )
            for i in range(4)
        ]

    def train_vial(self, vial: Vial, code: str, is_python: bool, balance: float):
        vial.status = "running"
        vial.code = code
        vial.codeLength = len(code)
        vial.isPython = is_python
        vial.wallet.balance = balance / 4
        return vial
