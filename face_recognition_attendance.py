import cv2
import os
import csv
import sys
import keyboard
from datetime import datetime
import numpy as np
from keras.models import Model, Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Activation, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input


# Global variables
epsilon1 = 0.23     # Cosine distance
epsilon2 = 90       # Euclidean distance
val_limit = 5       # Total number of pass
thre_limit = 4      # Number of pass for verification

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

model.load_weights('./model/vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def find_cosine_distance(database, test):
    a = np.matmul(np.transpose(database), test)
    b = np.sum(np.multiply(database, database))
    c = np.sum(np.multiply(test, test))
    cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    return cosine_distance


def find_euclidean_distance(database, test):
    euclidean_distance = database - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


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
        print(f"Image passes\t: {global_threshold}/{val_limit}")
    else:
        print("Unverified! They are not the same person!")
        print(f"Cosine Similarity\t: {cosine_similarity}")
        print(f"Euclidean Distance\t: {euclidean_distance}")
        print(f"Image passes\t: {global_threshold}/{val_limit}")

face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')


def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def record_attendance(username):
    # Generate current date, day, and timestamp
    current_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Read user information from user_info.csv
    user_info_path = './data/user_info.csv'
    with open(user_info_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] == username:
                matric_number = row[1]
                course = row[2]
                break
        else:
            print(f"User {username} not found in user_info.csv")
            return
    
    # Create a new CSV file for attendance recording if it doesn't exist
    attendance_filename = f'attendance_{current_date}.csv'
    attendance_path = os.path.join('./data/attendance', attendance_filename)

    # Check if the CSV file already exists
    file_exists = os.path.exists(attendance_path)

    fieldnames = ['Date', 'Timestamp', 'Username', 'Matric Number', 'Course']

    with open(attendance_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write attendance record to CSV
        writer.writerow({
            'Date': current_date,
            'Timestamp': timestamp,
            'Username': username,
            'Matric Number': matric_number,
            'Course': course
        })

    print(f"Attendance recorded for {username} on {current_date} at {timestamp}")


def search_database_image(username):
    user_db_dir = f'./data/anchor/{username}/'
    
    if not os.path.exists(user_db_dir):
        print(f"Directory '{user_db_dir}' does not exist")
        return []

    all_files = os.listdir(user_db_dir)
    return all_files

#
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else:
        print(f"Directory '{directory}' already exists")


def recognize_face(db_faces, username, frame, x, y, w, h):
    face_captured = f'./data/temp/{username}.jpg'
    
    val_count = 0
    for face in db_faces:
        full_dir = f'./data/anchor/{username}/{face}'
        verify_face(full_dir, face_captured, frame, username, x, y, w, h)
        val_count += 1
        print("------------------------------------------------------------------------------------------------")
        
        if val_count == val_limit:
            break

    if global_threshold >= thre_limit:
        record_attendance(username)
        #print("Verified. Attendance taken")
    else:
        print("Unverified. Attendance not taken.")


def save_face_img(username, directory, face):
    img_path = os.path.join(directory, f'{username}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path [12:]}'")
    print("------------------------------------------------------------------------------------------------")


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
                print("Press 'Esc' or 'q' to close camera...\n")
        
        cv2.imshow('Face Recognition', frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    global global_threshold
    
    while True:
        global_threshold = 0
        
        app_title = "Facial Data & Info Registration System"
        print("------------------------------------------------------------------------------------------------")
        print(app_title.center(96))
        print("------------------------------------------------------------------------------------------------\n")
        username = str(input('Enter Your Name (Eg. Lan Yi Xian)\t: '))
        temp_dir = './data/temp'
        
        print("")
        mkdir(temp_dir)
        webcam = initialize_webcam()
        print("Press 'C' to perform face verification...\n")
        face_valid_capture(webcam, temp_dir, username)
        print("Press 'Esc' or 'q' again to quit or any other key to continue...\n\n\n")
        
        while True:
            key = keyboard.read_event(suppress=True).name
            if key == 'q' or key == 'esc':
                sys.exit()
            else:
                break


if __name__ == "__main__":
    main()
