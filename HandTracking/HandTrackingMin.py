import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # Tells if it detects a hand or not
    #print(results.multi_hand_landmarks)

    # If there is a hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Shows the number for every part of the hand
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                # Multiplying x and y hand parts coordinates by their weights and heights
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                # Detecting one of the landmarks (only one finger, for example)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            # Shows hands' lines
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), 
               cv2.FONT_HERSHEY_PLAIN, 3, 
               (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)