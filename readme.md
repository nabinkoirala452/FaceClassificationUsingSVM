# Face Recognition with SVM Classifier

This project implements a real-time face recognition system using SVM (Support Vector Machine) classifier. It uses the `face_recognition` library for face encodings and `scikit-learn` for training the SVM model.

## Features
- Trains an SVM classifier to recognize faces from images.
- Supports saving and loading the trained model for future use.
- Allows retraining of the model when required.
- Real-time face recognition using a webcam.
- Identifies faces and displays the recognized name on the video feed.

## Dependencies
The following Python libraries are required:
- `os`
- `cv2` (OpenCV)
- `numpy`
- `face_recognition`
- `scikit-learn`
- `joblib`

You can install the required libraries using:
```bash
pip install opencv-python numpy face_recognition scikit-learn joblib

## How It Works
### Training:
- If a pre-trained model exists, you will be prompted to retrain or use the existing model.
- For training, it extracts face encodings from the images in the `images/` folder and maps them to the corresponding labels (person names).
- Saves the trained SVM model and label encoder for later use.

### Real-Time Recognition:
- Captures frames from the webcam.
- Detects faces in the frame, encodes them, and predicts the person using the trained SVM model.
- Draws a rectangle around the recognized face and displays the name.
