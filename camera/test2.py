from keras.models import load_model
import cv2
import numpy as np

file_name = "/Users/raymondchung/Documents/CS/project/image-recognition/emotion_model.keras"
loaded = load_model(file_name)

loaded.summary()

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_dict = {}
for i in range(len(emotions)):
    emotion_dict[i] = emotions[i]

cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        print(ret)
    # Create a face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(frame, 
                                               scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_color_frame = frame[y:y + h, x:x + w]
        rgb_frame = cv2.cvtColor(roi_color_frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_frame, (90, 90))
        normalized_img = resized_img.astype('float32') / 255.0
        cropped_img = np.expand_dims(normalized_img, axis=0)

        # predict the emotions
        emotion_prediction = loaded.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()