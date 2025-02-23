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
        self.image_size = image_size
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.1),
        ])
        self.model = self.create_model()
        
    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        x = self.data_augmentation(inputs)  # Built-in augmentation
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def load_images(self, directory, label):
        images = []
        labels = []
        valid_exts = ('.jpg', '.jpeg', '.png')
        
        for fname in os.listdir(directory):
            if fname.lower().endswith(valid_exts):
                try:
                    img = cv2.imread(os.path.join(directory, fname))
                    if img is None:
                        continue
                        
                    # Improved preprocessing pipeline
                    img = cv2.resize(img, self.image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess_input(img)  # Correct VGG16 preprocessing
                    
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        return np.array(images), np.array(labels)

    def prepare_data(self, pos_dir, neg_dir):
        pos_images, pos_labels = self.load_images(pos_dir, 1)
        neg_images, neg_labels = self.load_images(neg_dir, 0)

        if len(pos_images) == 0 or len(neg_images) == 0:
            raise ValueError("Check your directories - one class has no images!")

        X = np.concatenate([pos_images, neg_images])
        y = np.concatenate([pos_labels, neg_labels])
        
        # Stratified split with shuffling
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
        )
        return (X_train, y_train), (X_val, y_val)

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        batch_size = 32
        if len(X_train) < 100:
            batch_size = max(4, len(X_train) // 4)
            
        # Enhanced class weighting
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = {i:w for i,w in enumerate(class_weights)}
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save_model(self, path="my_picture_detector_model.keras"):
        self.model.save(path)
        print(f"Model saved at {path}")

    def load_model(self, path="my_picture_detector_model.keras"):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        
    def predict(self, image_path):
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
    detector = MyPictureDetector()
    
    try:
        # Update these paths to your actual directories
        (X_train, y_train), (X_val, y_val) = detector.prepare_data(
            "data/raw/my_pictures",
            "data/raw/not_my_pictures"
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Class distribution: {np.unique(y_train, return_counts=True)}")
        
        history = detector.train(X_train, y_train, X_val, y_val)
        detector.save_model("updated_detector.keras")
        
        # Plot training history
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()