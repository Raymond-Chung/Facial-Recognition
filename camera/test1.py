"""
credits: https://www.geeksforgeeks.org/python/face-detection-using-python-and-opencv-with-webcam/
"""

import cv2
face_cascae = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# open webcam and check for access
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error cam no work")
    exit()

# capture frames, convert to grayscale, and detect faces
# face detection works better on greyscale images
while True:
    ret, frame = cap.read()
    if not ret:
        print("error can't read frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    # draw rectangle around detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)
        
        cv2.putText(frame, "sup", (50, 150),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 255), 2, cv2.LINE_AA)   

        cv2.imshow('Face Detection', frame)

    # exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# relesae cam and destroy windows
cap.release()
cv2.destroyAllWindows()