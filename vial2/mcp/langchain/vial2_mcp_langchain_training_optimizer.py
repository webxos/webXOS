import torch.optim as optim

class TrainingOptimizer:
    def optimize(self, config):
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        return {"optimizer": optimizer, "learning_rate": config.learning_rate * 0.9}