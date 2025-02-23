import cv2
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self, config):
        self.config = config
        self.image_size = tuple(config['model']['image_size'])

    def process_images(self, source_dir, dest_dir):
        """Process and resize images from source to destination directory"""
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        for img_path in Path(source_dir).glob('*.[jJ][pP][gG]'):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resized = cv2.resize(img, self.image_size)
                    dest_path = Path(dest_dir) / img_path.name
                    cv2.imwrite(str(dest_path), resized)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def prepare_dataset(self):
        """Prepare the complete dataset for training"""
        # Implementation of dataset preparation
        pass

