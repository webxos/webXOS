import torch
import torch.nn as nn

class VialAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def initialize_vials():
    return [
        {
            "id": f"vial{i+1}",
            "model": VialAgent(),
            "status": "stopped",
            "code": "",
            "wallet": {"address": None, "balance": 0.0},
            "tasks": []
        }
        for i in range(4)
    ]

def export_vial(vial, filename):
    torch.save(vial['model'].state_dict(), f"{filename}.pt")
    with open(f"{filename}.md", 'w') as f:
        f.write(f"# Vial Agent: {vial['id']}\n- Status: {vial['status']}\n- Wallet Balance: {vial['wallet']['balance']}\n- Tasks: {', '.join(vial['tasks'])}\n```python\n{vial['code']}\n```")
