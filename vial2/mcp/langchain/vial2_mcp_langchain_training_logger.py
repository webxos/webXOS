import logging

class TrainingLogger:
    def __init__(self):
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        self.logger.info(message)