import cv2
import os

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def mkdir(username):
    parent_dir = './data/test_dataset'
    new_dir = os.path.join(parent_dir, username)
    
    # Create directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Directory '{username}' created successfully in '{parent_dir}'")
    else:
        print(f"Directory '{username}' already exists in '{parent_dir}'")


def save_face_img(face, count, username):
    file_name_path = f'./data/test_dataset/{username}/{str(count)}.jpg'
    cv2.imwrite(file_name_path, face)
    print(f"Successfully saved to {file_name_path}")


def face_capture(capture, username, sample_size):
    count = 0
    
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_locate:
            face_crop = frame[y:y + h, x:x + w]
            
            if face_crop is not None:
                count += 1
                face = cv2.resize(face_crop, (400, 400))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                save_face_img(face, count, username)
                cv2.imshow("Camera", frame)
            else:
                print("No face is detected")
        
        esc_key = cv2.waitKey(1)
        if count == sample_size or esc_key == ord('q') or esc_key == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    username = str(input("Enter your username : "))
    sample_size = int(input("Number of samples: "))
    mkdir(username)
    
    print("Webcam initializing...")
    webcam = initialize_webcam()
    
    print("Collecting data...")
    print("--------------------------------")
    
    face_capture(webcam, username, sample_size)
    
    print("--------------------------------")
    print("Data collection completed")


if __name__ == "__main__":
    main()
