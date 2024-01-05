import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerCounting/fingers'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for png in myList:
    image = cv2.imread(f'{folderPath}/{png}')
    overlayList.append(image)

pTime = 0
detector = htm.HandDetector(detectionConfidence=0.75)

while True:
    success, img = cap.read()

    # Each image is 64x64
    h, w, c = overlayList[0].shape
    img[10:h, 10:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)