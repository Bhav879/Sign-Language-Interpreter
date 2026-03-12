import cv2
import mediapipe as mp


#For MediaPipe 0.10.32+
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

#Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit()

print("The camera is ready")
print("\nShow hands in the box")
print("Press Esc to quit\n")

#Create hand landmarker (NEW API)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

#Simple detection using rectangle - Increases ease on user interface
while True:
    success, frame = cap.read()
    if not success:
        break
    
    #Mirroring the webcam
    frame = cv2.flip(frame, 1)
    
    #Box to show the user the hand detection area
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), 3)
    cv2.putText(frame, "SIGN HERE", (w//4, h//4-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    #Information for use
    cv2.putText(frame, "Press Esc to quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #Creator/Coder
    cv2.putText(frame, "by Bhavya", (w-120, h-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    #Show on screen
    cv2.imshow('Sign Language Interpreter', frame)
    
    #Quit
    if cv2.waitKey(1) & 0xFF == 27: #27 is the 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()

