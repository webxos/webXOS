import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class VialAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def run_agent(code: str):
    try:
        model = VialAgent()
        logger.info("Agent 3 initialized")
        return {"status": "running", "output": "Agent 3 initialized"}
    except Exception as e:
        logger.error(f"Agent 3 error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-11T00:44:00Z]** Agent 3 error: {str(e)}\n")
        raise
