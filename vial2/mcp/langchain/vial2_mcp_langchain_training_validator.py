import torch

class TrainingValidator:
    def validate(self, model, data):
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            return all(outputs == data)