import cv2
import mediapipe as mp
import urllib.request
import os

# Download required hand model if needed
model_file = "hand_landmarker.task"
if not os.path.exists(model_file):
    print("Downloading hand model")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_file)
    print("Download complete")

# Setup NEW MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_file)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# Camera
cap = cv2.VideoCapture(0)
print("\nPress Esc to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Detect
    result = detector.detect(mp_image)
    
    # Draw landmarks (21 points per hand)
    if result.hand_landmarks:
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            # Draw each landmark
            for i, landmark in enumerate(landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # Draw point
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green dots
                
                # Label first point
                if i == 0:  # Wrist
                    cv2.putText(frame, f"Hand {hand_index+1}", (x-20, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw connections (simplified - wrist to index)
            if len(landmarks) >= 9:
                # Wrist to index
                wrist = (int(landmarks[0].x * frame.shape[1]), 
                         int(landmarks[0].y * frame.shape[0]))
                index = (int(landmarks[5].x * frame.shape[1]), 
                         int(landmarks[5].y * frame.shape[0]))
                cv2.line(frame, wrist, index, (255, 0, 0), 2)
        
        cv2.putText(frame, "Landmarks detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'Esc' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "Show your hand to camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Sign Language Interpreter', frame)
    
    if cv2.waitKey(1) & 0xFF == 27: #27 is the ASCII for 'Esc'
        #Stopping program
        break

cap.release()
cv2.destroyAllWindows()
print("Hand detected with landmarks!")
