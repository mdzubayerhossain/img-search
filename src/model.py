import tensorflow as tf
from tensorflow.keras import layers, models

class PosterDetectionModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        """Build the model architecture"""
        model = models.Sequential([
            # Model architecture as defined earlier
            # ...
        ])
        return model