import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Disable scientific notation for easier understanding
np.set_printoptions(suppress=True)

# Load trained model (data obtained using Teachable Machine)
model = load_model("keras_model.h5", compile=False)

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(' ')[-1] for line in f.readlines()]

print("Press ESC to quit\n")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    
    # Converting image into required model (224x224) as expected by Teachable Machine
    img = cv2.resize(frame, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    letter = class_names[class_index]
    
    # Show result on frame
    if confidence > 0.7:  # Only show if confident in predictions
        text = f"{letter} ({confidence:.2f})"
        color = (0, 255, 0)  # Green
    else:
        text = "No letter detected"
        color = (0, 0, 255)  # Red
    
    # Text
    cv2.putText(frame, text, (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Confidence bar
    bar_x = 30
    bar_y = 100
    bar_width = int(200 * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + 20), (255, 255, 255), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)
    
    # Instructions
    cv2.putText(frame, "ESC to quit", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Display
    cv2.imshow('Sign Language Detector', frame)
    
    # Esc to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Successfully detected letters")
