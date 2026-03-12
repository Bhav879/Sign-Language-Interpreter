I built this interpreter with the aim of using something as complex and structure and mesmerising as code, for something meaningful, to make a difference. With no formal training in Computer Vision, I taught myself OpenCV (hands on learning as I made a fac counter) and Mediapipe.

Current working: It detects five letters - A, C, I, L, W 

Future: To detect and interpret the alphabet of sign language, maybe even a feature to detect different sign languages such as ISL and ASL.


#Libraries used: 
- OpenCV for webcam access
- MediaPipe for landmark detection
- TensorFlow with Keras to run a trained model that recognises chosen letters (Trained a model using Google's Teachable Machines)
- NumPy to help process image data before comparing it with the model


#Working:
MediaPipe displays 21 landmarks on each hand, such as on the wrists and through fingers. These are drawn/represented as green dots on the screen, and data obtained by tracking them is fed to the model, to compare with known data from over 150 samples for each letter, thus detecting the letter shown by user.


#Files:
- 'signlang_initialtemplate.py'  -  One of the first versions of detector with webcam feature and a defined space for hand detection.
- 'signlang_withlandmarks.py'  -  Version with mediapipe to detect landmarks as green dots when hands in front of webcam.
- 'signlang_detectletters.py'  -  Final version with the main detection of the five chosen letters using trained model.
- 'keras_model.h5' and 'labels.txt'  -  Trained model files
- 'LetterDetectionTest.mp4' - Demo video of working letter detection
