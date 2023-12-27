import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained VGG16 model for person recognition
# model = load_model('multiple_person_recognition_model.h5')
model = load_model('fine_tuned_face_recognition_model.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

# Load class indices from the training set
class_indices = {
    'Jia Min': 0,
    'Lan Yi Xian': 1,
    'Sin Khai': 2,
    # Add more class indices as per your training set
}

# Function to perform person recognition on a given face
def recognize_person(face):
    # Preprocess the face image
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0  # Normalize the pixel values

    # Make predictions using the loaded model
    predictions = model.predict(face)

    # Get the index of the predicted class (person)
    predicted_person_index = np.argmax(predictions)

    # Get the class label (person name) based on the index
    person_name = [k for k, v in class_indices.items() if v == predicted_person_index][0]

    return person_name

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Perform person recognition on the extracted face
        person_name = recognize_person(face_roi)

        # Draw a rectangle around the face and display the person's name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Person: {person_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
