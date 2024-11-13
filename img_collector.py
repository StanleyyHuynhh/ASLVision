import os
import cv2

# Directory to save data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (folders) and images per folder
number_of_classes = 24
dataset_size = 100

# Prompt the user to enter the class number they want to update
print("Available classes (folders):", list(range(number_of_classes)))
folder_to_update = int(input("Enter the folder number (0-23) you want to update: "))

# Verify the folder number is valid
if folder_to_update < 0 or folder_to_update >= number_of_classes:
    print("Invalid folder number. Please enter a number between 0 and 23.")
else:
    # Ensure the specified folder exists
    if not os.path.exists(os.path.join(DATA_DIR, str(folder_to_update))):
        os.makedirs(os.path.join(DATA_DIR, str(folder_to_update)))

    print('Collecting data for class {}'.format(folder_to_update))

    # Start the camera capture
    cap = cv2.VideoCapture(1)

    # Wait for user readiness
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture images for the specified folder
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(folder_to_update), '{}.jpg'.format(counter)), frame)
        counter += 1

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Data collection for class {folder_to_update} completed.")
