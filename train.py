import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

class MyPictureDetector:
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the detector with the specified image size and create the model.
        
        Args:
            image_size (tuple): The size to which images will be resized (width, height).
        """
        self.image_size = image_size
        self.model = self.create_model()
        
    def create_model(self):
        """
        Create a neural network model using transfer learning with VGG16.
        
        Returns:
            model: Compiled Keras model.
        """
        # Load pre-trained VGG16 model without the top layers
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
        base_model.trainable = False  # Freeze the base model layers

        # Add custom layers on top
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_images(self, directory, label):
        """
        Load and preprocess images from the specified directory with the given label.
        
        Args:
            directory (str): Path to the directory containing images.
            label (int): Label to assign to all images (1 for my pictures, 0 for not my pictures).
        
        Returns:
            images (np.array): Array of preprocessed images.
            labels (np.array): Array of corresponding labels.
        """
        images = []
        labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png')

        print(f"Loading images from {directory} with label {label}")
        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(directory, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        print(f"Warning: Skipping corrupt image {filename}")
                        continue
                    
                    img = cv2.resize(img, self.image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess_input(img)  # Apply VGG16 preprocessing
                    
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Successfully loaded {len(images)} images from {directory}")
        return np.array(images), np.array(labels)

    def prepare_data(self, my_pictures_dir, not_my_pictures_dir):
        """
        Prepare the dataset by loading images and splitting into training and validation sets.
        
        Args:
            my_pictures_dir (str): Path to directory containing "my pictures."
            not_my_pictures_dir (str): Path to directory containing "not my pictures."
        
        Returns:
            (X_train, y_train), (X_val, y_val): Training and validation datasets.
        """
        my_images, my_labels = self.load_images(my_pictures_dir, 1)
        not_my_images, not_my_labels = self.load_images(not_my_pictures_dir, 0)

        if len(my_images) == 0 or len(not_my_images) == 0:
            raise ValueError("One of the directories has no valid images.")

        X = np.concatenate([my_images, not_my_images])
        y = np.concatenate([my_labels, not_my_labels])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return (X_train, y_train), (X_val, y_val)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train the model with data augmentation and class weighting.
        
        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels.
            epochs (int): Number of epochs to train for.
        
        Returns:
            history: Training history.
        """
        batch_size = min(32, len(X_train) // 2)  # Ensure at least 2 batches
        print(f"Using batch size: {batch_size}")
        
        # Compute class weights to handle imbalance
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=classes, y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Class weights: {class_weights}")
        
        # Data augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Convert to TensorFlow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).map(
            lambda x, y: (data_augmentation(x), y)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.01
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
            )
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def save_model(self, path="my_picture_detector_model.keras"):
        """
        Save the trained model to the specified path.
        
        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)
        print(f"Model saved at {path}")

    def load_model(self, path="my_picture_detector_model.keras"):
        """
        Load a trained model from the specified path.
        
        Args:
            path (str): Path to the saved model.
        """
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        
    def predict(self, image_path):
        """
        Predict whether the given image is "my picture" or not.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            bool: True if the image is "my picture," False otherwise. Returns None on error.
        """
        try:
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                raise ValueError("Invalid image file")
            
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img)  # Apply VGG16 preprocessing
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            prediction = self.model.predict(img)[0][0]
            return prediction > 0.5  # True if it's my picture
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

def main():
    """
    Main function to prepare data, train the model, and save it.
    """
    my_pictures_dir = "data/raw/my_pictures"
    not_my_pictures_dir = "data/raw/not_my_pictures"
    
    if not os.path.exists(my_pictures_dir) or not os.path.exists(not_my_pictures_dir):
        print("Error: Data directories not found. Please create: ")
        print(f"- {my_pictures_dir}")
        print(f"- {not_my_pictures_dir}")
        return
    
    detector = MyPictureDetector()
    
    try:
        print("Preparing dataset...")
        (X_train, y_train), (X_val, y_val) = detector.prepare_data(
            my_pictures_dir, not_my_pictures_dir
        )
        
        print("\nStarting training...")
        history = detector.train(X_train, y_train, X_val, y_val)
        
        detector.save_model("my_picture_detector_model.keras")
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main()