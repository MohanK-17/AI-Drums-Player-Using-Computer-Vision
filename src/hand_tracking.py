# import cv2 
# import mediapipe as mp
# import time

# cmap=cv2.VideoCapture(0)

# mpHands=mp.solutions.hands
# hands=mpHands.Hands()
# mpDraw=mp.solutions.drawing_utils

# pTime=0
# cTime=0


# while True:
#     success, img=cmap.read()
#     imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results=hands.process(imgRGB)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
#                 mp.solutions.drawing_utils.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
p_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("AI Drums - Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
