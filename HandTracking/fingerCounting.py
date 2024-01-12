import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'fingers'
myList = os.listdir(folderPath)
# print(myList)
myList.sort()
overlayList = []
for png in myList:
    image = cv2.imread(f'{folderPath}/{png}')
    overlayList.append(image)

print(myList)

pTime = 0
detector = htm.HandDetector(detectionConfidence=0.75)
# Number of the fingers, in order, starting with the thumb
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    # print(lmlist)

    if len(lmlist) != 0:
        fingers = []
        # Checking the x-axis of the thumb with one value below it
        if lmlist[tipIds[0]][1] < lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for loop for all of the fingers, except the thumb
        for id in range(1, 5):
            # Index of the finger, gets the y-axis pixels (height, number 2)
            # It checks if the finger is closed or open based on the position
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        # Each image is 64x64
        h, w, c = overlayList[0].shape
        # h+10 because it starts at 10
        img[10:h+10, 10:w+10] = overlayList[totalFingers]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)