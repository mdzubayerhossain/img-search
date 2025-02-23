# My Picture Detector

This project is a binary image classifier that detects whether an image is "my picture" or "not my picture." It uses transfer learning with the VGG16 model and is implemented in TensorFlow/Keras.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Training the Model](#training-the-model)
5. [Testing the Model](#testing-the-model)
6. [Directory Structure](#directory-structure)
7. [License](#license)

---

## Project Overview
The goal of this project is to classify images into two categories:
- **My Picture**: Images that belong to a specific category (e.g., your personal photos).
- **Not My Picture**: Images that do not belong to that category.

The model is trained using transfer learning with the VGG16 architecture and fine-tuned for binary classification.

---

## Requirements
To run this project, you need the following:
- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- NumPy
- scikit-learn
- Matplotlib (for visualization)

You can install the required packages using the following command:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
