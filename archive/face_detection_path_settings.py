import cv2
import os
import uuid
import sys
import keyboard

# Load HAAR Classifier
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def mkdir(username, collection_path):
    if collection_path == 0:
        parent_dir = r'data\anchor'
    elif collection_path == 1:
        parent_dir = r'data\positive'
    else:
        raise ValueError("Invalid collection path")
    
    new_dir = os.path.join(parent_dir, username)
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"\nDirectory '{username}' created successfully in '{parent_dir}'")
    else:
        print(f"\nDirectory '{username}' already exists in '{parent_dir}'")
    
    return new_dir


def save_face_img(new_dir, face, count, sample_size):
    img_path = os.path.join(new_dir, f'{uuid.uuid1()}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path [5:]}' _______________ [{count}/{sample_size}]")


def face_capture(capture, new_dir, sample_size):
    count = 0
    
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        '''if len(face_locate) == 0:
            print("No face is detected")'''
        
        for (x, y, w, h) in face_locate:
            face_crop = frame[y:y + h, x:x + w]
            
            if face_crop is not None:
                count += 1
                face = cv2.resize(face_crop, (400, 400))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(count), (x + w - 50, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                save_face_img(new_dir, face, count, sample_size)
        
        cv2.imshow("Camera", frame)
        esc_cam = cv2.waitKey(1)
        if count == sample_size or esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    while True:
        project_topic = "Facial Recognition Smart Attendance System"
        print("------------------------------------------------------------------------------------------------")
        print(project_topic.center(96))
        print("------------------------------------------------------------------------------------------------\n")
        
        username = str(input("Enter your username (Eg. Lan Yi Xian)\t\t: "))
        sample_size = int(input("Enter number of face samples required\t\t: "))
        collection_path = int(input("Select directory ([0] Anchor / [1] Positive)\t: "))
        
        new_dir = mkdir(username, collection_path)
        
        print("Web camera initializing...")
        webcam = initialize_webcam()
        
        print("\nData collection starting...")
        print("------------------------------------------------------------------------------------------------")
        
        face_capture(webcam, new_dir, sample_size)
        
        print("\n------------------------------------------------------------------------------------------------")
        print(f"{sample_size} face samples collected and stored in the directory: '{new_dir}'")
        print("------------------------------------------------------------------------------------------------")
        print("Data collection completed")
        print("Press 'Esc' or 'q' to quit or any other key to continue...\n\n\n")
        
        while True:
            key = keyboard.read_event(suppress=True).name
            if key == 'q' or key == 'esc':
                sys.exit()
            else:
                break


if __name__ == "__main__":
    main()