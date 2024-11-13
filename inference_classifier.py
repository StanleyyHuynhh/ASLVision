import pickle
import cv2
import mediapipe as mp
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the trained model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Initialize Mediapipe for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode= True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Label dictionary for classes (update this based on your dataset)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

# Start video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    data_aux = []
    h, w, _ = frame.shape  # Frame dimensions for scaling landmarks

    # Check for detected hand landmarks
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        x_min, y_min, x_max, y_max = w, h, 0, 0  # Initialize bounding box coordinates

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract coordinates for bounding box and landmarks
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                data_aux.extend([landmark.x, landmark.y])

                # Update bounding box coordinates
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Predict the gesture if we have exactly 42 features
        if len(data_aux) == 42:
            prediction = model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]

            # Display the prediction on the video feed
            cv2.putText(frame, f"Prediction: {predicted_character}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video frame with landmarks, bounding box, and predictions
    cv2.imshow("ASL Sign Prediction", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()