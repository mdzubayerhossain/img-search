import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MyPictureDetector:
    def __init__(self, model_path, image_size=(224, 224)):
        """
        Initialize the detector with the trained model and image size.
        
        Args:
            model_path (str): Path to the saved model file.
            image_size (tuple): Size to which images will be resized (width, height).
        """
        self.image_size = image_size
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def preprocess_image(self, image_path):
        """
        Preprocess the image for prediction.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            np.array: Preprocessed image ready for prediction.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to load image from {image_path}")
        
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.keras.applications.vgg16.preprocess_input(img)  # VGG16 preprocessing
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def predict(self, image_path):
        """
        Predict whether the image is "my picture" or not.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            bool: True if the image is "my picture," False otherwise.
        """
        try:
            img = self.preprocess_image(image_path)
            prediction = self.model.predict(img)[0][0]
            return prediction > 0.5  # True if it's my picture
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

    def display_image(self, image_path, prediction):
        """
        Display the image with the prediction result.
        
        Args:
            image_path (str): Path to the image file.
            prediction (bool): Prediction result (True/False).
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img)
        plt.title(f"Prediction: {'My Picture' if prediction else 'Not My Picture'}")
        plt.axis('off')
        plt.show()

def main():
    # Path to the saved model
    model_path = "updated_detector.keras"  # Update this if your model has a different name
    
    # Path to the test image
    test_image_path = r"F:\Coding\project\poster_detection_project\tes3.png"  # Replace with the path to your test image
    
    # Initialize the detector
    detector = MyPictureDetector(model_path)
    
    # Make a prediction
    prediction = detector.predict(test_image_path)
    
    if prediction is not None:
        print(f"Prediction: {'My Picture' if prediction else 'Not My Picture'}")
        
        # Display the image with the prediction
        detector.display_image(test_image_path, prediction)
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main()