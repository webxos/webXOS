import torch
import torch.nn as nn
import logging
import re

class MCPClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def forward(self, x):
        try:
            return torch.sigmoid(self.fc(x))
        except Exception as e:
            self.logger.error(f"Forward pass error: {str(e)}")
            return torch.zeros_like(x)

    def train(self, input_data, code):
        try:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            target = torch.ones(input_data.size(0), 1)
            optimizer.zero_grad()
            output = self(input_data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            self.send_gmails(code)
            self.logger.info("MCP Client trained successfully")
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")

    def send_gmails(self, code):
        try:
            if 'send_gmails' in code.lower():
                matches = re.findall(r'send_gmails\s*\(\s*["\'](.*?)["\']\s*\)', code, re.IGNORECASE)
                self.logger.info(f"Send gmails simulated with recipients: {matches}")
            else:
                self.logger.info("No send_gmails command found in code")
        except Exception as e:
            self.logger.error(f"Send gmails error: {str(e)}")
