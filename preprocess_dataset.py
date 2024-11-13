import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Directory where images are saved
DATA_DIR = './data'
data = []
labels = []

# Loop over each class folder and process each image
for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        data_aux = []
        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(img_rgb)

        # Only proceed if exactly one hand with full landmarks is detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y])

            # Append only samples with exactly 42 features
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(label)

# Save data and labels to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Data processing complete with consistent feature count.")
