import os
import cv2
import numpy as np
import face_recognition
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import joblib  # To save and load model files
 
# Paths to model files
model_path = 'svm_model.joblib'
label_encoder_path = 'label_encoder.joblib'


# Check if model and label encoder already exist
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    # Prompt user for retraining
    retrain = input("A pre-trained model exists. Do you want to retrain? (y/n): ").strip().lower()
    if retrain == 'y':
        # Proceed with training
        encoded_faces = []
        class_names = []
        
        # Define the path to the main images folder
        image_folder_path = 'images'
        
        # Loop through each subfolder (each person) in the main folder
        for person_name in os.listdir(image_folder_path):
            person_folder = os.path.join(image_folder_path, person_name)
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            encoded_faces.append(encodings[0])
                            class_names.append(person_name)
        
        # Encode class labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(class_names)

        # Train SVM classifier
        svm_clf = svm.SVC(kernel='linear', probability=True)
        svm_clf.fit(encoded_faces, labels)
        print("Training complete with SVM classifier")

        # Save model and label encoder
        joblib.dump(svm_clf, model_path)
        joblib.dump(label_encoder, label_encoder_path)
        print("Model and label encoder saved.")
    else:
        # Load existing model and label encoder
        svm_clf = joblib.load(model_path)
        label_encoder = joblib.load(label_encoder_path)
        print("Loaded existing SVM model and label encoder.")
else:
    # If no model exists, proceed with training
    print("Training in process")
    encoded_faces = []
    class_names = []
    
    image_folder_path = 'images'
    
    for person_name in os.listdir(image_folder_path):
        person_folder = os.path.join(image_folder_path, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # encodings = face_recognition.face_encodings(image)
                    try:

                        encodings = face_recognition.face_encodings(image)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue  # Skip this image and proceed with others

                    if encodings:
                        encoded_faces.append(encodings[0])
                        class_names.append(person_name)
    
    # Encode class labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(class_names)

    # Train SVM classifier
    svm_clf = svm.SVC(kernel='linear', probability=True)
    svm_clf.fit(encoded_faces, labels)
    print("Training complete with SVM classifier")

    # Save model and label encoder
    joblib.dump(svm_clf, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    print("Model and label encoder saved.")



# Initialize video capture for real-time face recognition
video_capture = cv2.VideoCapture(0)

while True:
    success, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_encoding = face_encoding.reshape(1, -1)
        probabilities = svm_clf.predict_proba(face_encoding)
        best_match_index = np.argmax(probabilities)
        max_probability = probabilities[0][best_match_index]
        if max_probability > 0.5:
            name = label_encoder.inverse_transform([best_match_index])[0]
        else:
            name = "UnKnown"

        # Draw rectangle around face and display name
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (153, 153, 7), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (106, 45, 87), cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (195, 229, 232), 2)

    cv2.imshow('webcam', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    if cv2.getWindowProperty('webcam', cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()

cv2.destroyAllWindows()

