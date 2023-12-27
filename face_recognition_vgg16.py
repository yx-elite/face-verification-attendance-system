import cv2
import os
import numpy as np
from keras.models import load_model


model = load_model('model/face_recognition_model.h5')
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def class_prediction():
    class_path = 'data/anchor'
    
    # Extract all classes from path and sort them
    classes = os.listdir(class_path)
    sorted_class = sorted(classes)
    
    # Create an empty dictionary
    class_indices = {}
    
    # Rearrange the placement of username and index
    for index, username in enumerate(sorted_class):
        class_indices[username] = index
    
    return class_indices


def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def recognize_face(face, class_indices):
    # Resize & preprocess for model compatibility
    face = cv2.resize(face, (224, 224))
    face = np.reshape(face, [1, 224, 224, 3])
    face = face * 1./255
    
    # Perform face prediction by model
    face_pred = model.predict(face)
    pred_index = np.argmax(face_pred)
    
    return pred_index


def face_valid_capture(capture, class_indices):
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_locate:
            face_crop = frame[y:y + h, x:x + w]
            pred_index = recognize_face(face_crop, class_indices)
            
            # Loop through dictionary and extract predicted person's name
            match = False
            for username, index in class_indices.items():
                if pred_index == index:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, str(username), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    match = True
                    break
            
            if not match:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "User not found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    webcam = initialize_webcam()
    class_indices = class_prediction()
    face_valid_capture(webcam, class_indices)


if __name__ == "__main__":
    main()