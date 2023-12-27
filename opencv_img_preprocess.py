import cv2
import os


# Load HAAR Classifier
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def search_all_images(raw_path):
    if not os.path.exists(raw_path):
        print(f"Directory '{raw_path}' does not exist")

    all_files = os.listdir(raw_path)

    return all_files


def mkdir(name):
    out_path = f'img\processed\{name}'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"\nDirectory '{out_path}' created successfully")
    else:
        print(f"\nDirectory '{out_path}' already exists")
    
    return out_path


def save_face_img(out_dir, name, i, face):
    processed = os.path.join(str(out_dir), f'{name}_{i}.jpg')
    cv2.imwrite(processed, face)
    print(f"Processed image: '{processed}'")


i = 1

def img_preprocess(raw_dir, out_dir, name):
    raw_images = search_all_images(raw_dir)
    
    for image in raw_images:
        img_full_path = f'{raw_dir}/{image}'
        img = cv2.imread(img_full_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        global i 
        for (x, y, w, h) in face_locate:
            face_crop = img[y:y + h, x:x + w]
            face = cv2.resize(face_crop, (400, 400))
            save_face_img(out_dir, name, i, face)
            i += 1


def main():
    name = 'Ji Chang Wook'
    raw_path = f'./img/raw/{name}'
    out_path = mkdir(name)
    img_preprocess(raw_path, out_path, name)



if __name__ == "__main__":
    main()