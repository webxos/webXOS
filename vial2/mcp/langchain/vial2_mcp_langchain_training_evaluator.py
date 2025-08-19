import torch

class TrainingEvaluator:
    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            return {"accuracy": (outputs == data).float().mean().item()}