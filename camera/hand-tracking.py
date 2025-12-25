import cv2
import mediapipe as mp 

cap = cv2.VideoCapture(0)

# Hand tracking
mp_hands = mp.solutions.hands # a module
hand = mp_hands.Hands() # actual object

# hand landmarks
mp_drawings = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hand.process(RGB_frame)

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("hands", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()