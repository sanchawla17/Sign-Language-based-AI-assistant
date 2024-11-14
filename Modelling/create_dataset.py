from pathlib import Path
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = 'C:\\Users\\Sanyam\\Documents\\Projects\\Multilingual-Sign-Language-AI-assistant\\Modelling\\test2\\data'
NUM_CLASSES = 16
NUM_LANDMARKS = 21

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
    return landmarks

data = []
labels = []

for j in range(NUM_CLASSES):
    class_dir = Path(DATA_DIR) / str(j)  # Use Pathlib for cross-platform compatibility
    if not class_dir.exists():
        print(f"Directory {class_dir} does not exist.")
        continue

    for img_path in class_dir.iterdir():  # Iterate through files in each class directory
        if img_path.suffix == '.jpg':  # Only process JPG images
            left_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
            right_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
            
            img = cv2.imread(str(img_path))  # Convert Path to str for OpenCV
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == "Left":
                        left_hand_landmarks = extract_landmarks(hand_landmarks)
                    elif handedness.classification[0].label == "Right":
                        right_hand_landmarks = extract_landmarks(hand_landmarks)

            features = np.concatenate([left_hand_landmarks, right_hand_landmarks])
            data.append(features)
            labels.append(j)

print(f"Collected {len(data)} samples.")

if data:
    with open('./data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Dataset created successfully!")
else:
    print("No data collected. Please check if the image directories contain images.")

hands.close()
