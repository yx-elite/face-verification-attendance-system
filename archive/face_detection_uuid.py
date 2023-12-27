import cv2
import os
import uuid

# Load HAAR Classifier
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def mkdir(username):
    parent_dir = r'data\test_dataset'
    new_dir = os.path.join(parent_dir, username)
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Directory '{username}' created successfully in '{parent_dir}'")
    else:
        print(f"Directory '{username}' already exists in '{parent_dir}'")

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                save_face_img(new_dir, face, count, sample_size)
        
        cv2.imshow("Camera", frame)
        esc_key = cv2.waitKey(1)
        if count == sample_size or esc_key == ord('q') or esc_key == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    username = str(input("Enter your username\t: "))
    sample_size = int(input("Number of samples\t: "))
    new_dir = mkdir(username)
    
    print("Webcam initializing...")
    webcam = initialize_webcam()
    
    print("Data collection starting...")
    print("--------------------------------")
    
    face_capture(webcam, new_dir, sample_size)
    
    print("--------------------------------")
    print("Data collection completed")


if __name__ == "__main__":
    main()