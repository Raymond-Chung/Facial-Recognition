from keras.models import load_model
import cv2
import numpy as np

file_name = "../image-recognition/emotion_model.keras"
loaded = load_model(file_name)

loaded.summary()

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_dict = {}
for i in range(len(emotions)):
    emotion_dict[i] = emotions[i]

monkey_dict = {
                'angry' : cv2.imread('../images/monkey_images/angry.jpg'),
                'fear' : cv2.imread('../images/monkey_images/fear.jpg'),
                'happy' : cv2.imread('../images/monkey_images/happy.jpg'),
                'neutral' : cv2.imread('../images/monkey_images/neutral.jpg'), 
                'sad' : cv2.imread('../images/monkey_images/sad.jpg'), 
                'surprise' : cv2.imread('../images/monkey_images/suprise.jpg')
                }

cap = cv2.VideoCapture(0)

# Create a face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Frame counter so its running every n frame
frame_count = 0
UPDATE_N_FRAME = 30

# Logic to keep box/text between updates
last_face = []
last_emotion = ""

first_frame = True

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        print(ret)

    # Proccess only at every n frame
    if frame_count % UPDATE_N_FRAME == 0: 

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(frame, 
                                               scaleFactor=1.3, minNeighbors=5)
        last_face = num_faces 

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_color_frame = frame[y:y + h, x:x + w]
            rgb_frame = cv2.cvtColor(roi_color_frame, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_frame, (90, 90))
            normalized_img = resized_img.astype('float32')
            cropped_img = np.expand_dims(normalized_img, axis=0)

            # predict the emotions
            emotion_prediction = loaded.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            last_emotion = emotion_dict[maxindex]
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    for (x, y, w, h) in last_face:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    frame_count += 1

    cv2.imshow('Emotion Detection', frame)

    if first_frame: 
        # Resizing and moving window (WINDOW_NORMAL allows for resizing)
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Emotion Detection', 0, 0)
        cv2.resizeWindow('Emotion Detection', 640, 360)
        first_frame = False
    
    monkey = monkey_dict.get(last_emotion)
    if monkey is not None:
        cv2.namedWindow('Monkey', cv2.WINDOW_NORMAL)
        cv2.imshow('Monkey', monkey)
        cv2.moveWindow('Monkey', 800, 0)
        # cv2.resizeWindow('Monkey', 640, 360)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 