import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC  # Import Support Vector Classifier

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
y = y.astype(int)
X = X / 255.0  # Normalize pixel values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a SVM model
svm_model = SVC(kernel='linear', random_state=42)  
svm_model.fit(X_train, y_train)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a smaller canvas for drawing (e.g., 300x300)
canvas = np.zeros((300, 300, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

drawing = False
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            # Map the hand coordinates to the smaller canvas
            canvas_x = int(x * (300 / w))
            canvas_y = int(y * (300 / h))

            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # Draw on canvas
            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (canvas_x, canvas_y), (255, 255, 255), 10)
                prev_x, prev_y = canvas_x, canvas_y
            else:
                prev_x, prev_y = None, None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display instructions
    cv2.putText(frame, "Press 'S' to Start, 'E' to Stop, 'Q' to Predict", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Air Writing", frame)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        drawing = True
        print("Drawing started.")
    elif key == ord('e'):
        drawing = False
        print("Drawing stopped.")
    elif key == ord('q'):  # Quit and predict the digit
        print("Quitting... Predicting the digit.")
        break

cap.release()
cv2.destroyAllWindows()

# Preprocess the canvas image for prediction
canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Increase brightness by scaling pixel values
scale_factor = 1.5  # Increase brightness by 30%
canvas_gray = np.clip(canvas_gray.astype(float) * scale_factor, 0, 255).astype(np.uint8)

# Step 1: Find the bounding box of the digit
_, thresh = cv2.threshold(canvas_gray, 127, 255, cv2.THRESH_BINARY_INV)  # Invert and threshold
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get the bounding box of the largest contour (the digit)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Add some padding to the bounding box
    padding = 10  # Add 10 pixels of padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(canvas_gray.shape[1] - x, w + 2 * padding)
    h = min(canvas_gray.shape[0] - y, h + 2 * padding)

    # Step 2: Crop the canvas to the bounding box
    cropped = canvas_gray[y:y+h, x:x+w]

    # Step 3: Resize the cropped image to 28x28 while preserving aspect ratio
    if h > w:
        resized = cv2.resize(cropped, (int(28 * w / h), 28), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(cropped, (28, int(28 * h / w)), interpolation=cv2.INTER_AREA)

    # Step 4: Pad the resized image to make it 28x28
    pad_top = (28 - resized.shape[0]) // 2
    pad_bottom = 28 - resized.shape[0] - pad_top
    pad_left = (28 - resized.shape[1]) // 2
    pad_right = 28 - resized.shape[1] - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
else:
    # If no contours are found, use the entire canvas
    padded = cv2.resize(canvas_gray, (28, 28), interpolation=cv2.INTER_AREA)

# Invert colors to match MNIST format (black digit on white background)
canvas_inverted = padded

# Normalize pixel values
canvas_normalized = canvas_inverted / 255.0

# Flatten the image into a 1D array
test_sample = canvas_normalized.reshape(1, -1)  # Shape: (1, 784)

# Predict the digit
predicted_digit = svm_model.predict(test_sample)

# Display the final drawn image and prediction
plt.figure(figsize=(6, 6))
plt.imshow(canvas_inverted, cmap='gray')
plt.axis('off')
plt.title(f"Predicted Digit: {predicted_digit[0]}")
plt.show()