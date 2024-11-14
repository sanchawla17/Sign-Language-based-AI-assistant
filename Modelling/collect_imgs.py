import os
import cv2
import time

DATA_DIR = './test2/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 100
capture_interval = 0.2

cap = cv2.VideoCapture(0)

for class_index in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_index))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_index}')
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Starting collection in 2 seconds...")
            time.sleep(2)
            break

    for img_num in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.putText(frame, f'Class {class_index} - Image {img_num + 1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        img_path = os.path.join(class_dir, f'{img_num}.jpg')
        cv2.imwrite(img_path, frame)

        print(f'Saved {img_path}')
        time.sleep(capture_interval)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
