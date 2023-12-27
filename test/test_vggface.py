import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from PIL import Image


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


# Pretained Weights: https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
from keras.models import model_from_json

model.load_weights('vgg_face_weights.h5')

# Define image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


# Cosine Similarity
def find_cosine_distance(database, test):
    a = np.matmul(np.transpose(database), test)
    b = np.sum(np.multiply(database, database))
    c = np.sum(np.multiply(test, test))
    cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    return cosine_distance


# Euclidean Similarity
def find_euclidean_distance(database, test):
    euclidean_distance = database - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance


# Define model input and output
vgg_face_descriptor = Model(inputs = model.layers[0].input, outputs = model.layers[-2].output)

epsilon1 = 0.23    # Cosine similarity
epsilon2 = 90      # Euclidean distance similarity

def verify_face(img1, img2, threshold, frame, username, x, y, w, h):
    db_rep = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    test_rep = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]

    cosine_similarity = find_cosine_distance(db_rep, test_rep)
    euclidean_distance = find_euclidean_distance(db_rep, test_rep)
    
    '''fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image.load_img(img1))
    plt.xticks([]); plt.yticks([])
    fig.add_subplot(1, 2, 2)
    plt.imshow(image.load_img(img2))
    plt.xticks([]); plt.yticks([])
    plt.show(block = True)'''

    if(cosine_similarity < epsilon1 and euclidean_distance < epsilon2):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(username), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print("verified... they are same person")
        print(f"Cosine Similarity\t: {cosine_similarity}")
        print(f"Euclidean Distance\t: {euclidean_distance}")
        threshold += 1
        print(f"Image passes\t: {threshold}/10")
    else:
        print("unverified! they are not same person!")
        print(f"Cosine Similarity\t: {cosine_similarity}")
        print(f"Euclidean Distance\t: {euclidean_distance}")
        print(f"Image passes\t: {threshold}/10")
    
    return threshold

face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return(capture)


# Loop through directory and return all images in a list
def search_database_image(username):
    user_db_dir = f'./data/anchor/{username}/'
    
    if not os.path.exists(user_db_dir):
        print(f"Directory '{user_db_dir}' does not exists")
    
    all_files = os.listdir(user_db_dir)
    
    return all_files


def mkdir():
    new_dir = './data/temp'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"\nDirectory '{new_dir}' created successfully")
    else:
        print(f"\nDirectory '{new_dir}' already exists")
    
    return new_dir


def recognize_face(db_faces, username, frame, x, y, w, h):
    face_captured = f'./data/temp/{username}.jpg'
    
    val_count = 0
    threshold = 0
    for face in db_faces:
        full_dir = f'./data/anchor/{username}/{face}'
        verify_face(full_dir, face_captured, threshold, frame, username, x, y, w, h)
        val_count += 1
        print('--------------------------------------------------------')
        
        if val_count == 10:
            break

    if threshold >= 8:
        print("Verified. Attendance taken")
    else:
        print("Unverified. Attendance not taken.")


def save_face_img(username, new_dir, face):
    img_path = os.path.join(new_dir, f'{username}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path [5:]}'")


def face_valid_capture(capture, new_dir, username):
    db_img_file = search_database_image(username)
    
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
                save_face_img(username, new_dir, face)
                recognize_face(db_img_file, username, frame, x, y, w, h)
        
        cv2.imshow('Face Recognition', frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    username = 'Lee Jia Min'
    new_dir = mkdir()
    webcam = initialize_webcam()
    face_valid_capture(webcam, new_dir, username)


if __name__ == "__main__":
    main()