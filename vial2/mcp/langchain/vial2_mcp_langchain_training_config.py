class TrainingConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)