import cv2
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

# Global variables
epsilon1 = 0.23    # Cosine similarity threshold
epsilon2 = 90      # Euclidean distance threshold
global_threshold = 0

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))


# Load pre-trained weights
model.load_weights('vgg_face_weights.h5')

# Face descriptor model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Function to preprocess an image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to find cosine distance
def find_cosine_distance(database, test):
    a = np.matmul(np.transpose(database), test)
    b = np.sum(np.multiply(database, database))
    c = np.sum(np.multiply(test, test))
    cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    return cosine_distance

# Function to find Euclidean distance
def find_euclidean_distance(database, test):
    euclidean_distance = database - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# Function to verify faces
def verify_face(img1, img2, frame, username, x, y, w, h):
    global global_threshold

    db_rep = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
    test_rep = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]

    cosine_similarity = find_cosine_distance(db_rep, test_rep)
    euclidean_distance = find_euclidean_distance(db_rep, test_rep)

    if cosine_similarity < epsilon1 and euclidean_distance < epsilon2:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(username), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print("Verified... they are the same person")
        print(f"Cosine Similarity\t: {cosine_similarity}")
        print(f"Euclidean Distance\t: {euclidean_distance}")
        global_threshold += 1
        print(f"Image passes\t: {global_threshold}/10")
    else:
        print("Unverified! They are not the same person!")
        print(f"Cosine Similarity\t: {cosine_similarity}")
        print(f"Euclidean Distance\t: {euclidean_distance}")
        print(f"Image passes\t: {global_threshold}/10")

face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

# Function to initialize the webcam
def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture

# Function to search for database images
def search_database_image(username):
    user_db_dir = f'./data/anchor/{username}/'
    
    if not os.path.exists(user_db_dir):
        print(f"Directory '{user_db_dir}' does not exist")
        return []

    all_files = os.listdir(user_db_dir)
    return all_files

# Function to create a directory if it doesn't exist
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else:
        print(f"Directory '{directory}' already exists")

# Function to recognize faces in the database
def recognize_face(db_faces, username, frame, x, y, w, h):
    face_captured = f'./data/temp/{username}.jpg'
    
    val_count = 0
    for face in db_faces:
        full_dir = f'./data/anchor/{username}/{face}'
        verify_face(full_dir, face_captured, frame, username, x, y, w, h)
        val_count += 1
        print('--------------------------------------------------------')
        
        if val_count == 10:
            break

    if global_threshold >= 8:
        print("Verified. Attendance taken")
    else:
        print("Unverified. Attendance not taken.")

# Function to save the captured face image
def save_face_img(username, directory, face):
    img_path = os.path.join(directory, f'{username}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path [12:]}'")
    print("--------------------------------------------------------")

# Function to capture a valid face
def face_valid_capture(capture, directory, username):
    db_img_files = search_database_image(username)
    
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_locate:
            face_crop = frame[y:y + h, x:x + w]
            face = cv2.resize(face_crop, (400, 400))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cap_cam = cv2.waitKey(1)
            if cap_cam == ord('c'):
                save_face_img(username, directory, face)
                recognize_face(db_img_files, username, frame, x, y, w, h)
        
        cv2.imshow('Face Recognition', frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    global global_threshold
    username = 'Lee Jia Min'
    temp_dir = './data/temp'
    mkdir(temp_dir)
    webcam = initialize_webcam()
    face_valid_capture(webcam, temp_dir, username)

if __name__ == "__main__":
    main()
