import torch
import os

class TrainingCheckpoint:
    def __init__(self, checkpoint_dir="/app/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, epoch):
        torch.save(model.state_dict(), f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")

    def load(self, model, epoch):
        return torch.load(f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")