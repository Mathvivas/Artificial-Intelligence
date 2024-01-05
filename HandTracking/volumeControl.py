import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import subprocess

def get_volume():
	proc = subprocess.Popen('/usr/bin/amixer sget Master', shell=True, stdout=subprocess.PIPE)
	amixer_stdout = str(proc.communicate()[0],'UTF-8').split('\n')[4]
	proc.wait()
	find_start = amixer_stdout.find('[') + 1
	find_end = amixer_stdout.find('%]', find_start)
	return float(amixer_stdout[find_start:find_end])

def set_volume(volume):
	val = volume
	val = float(int(val))
	proc = subprocess.Popen('/usr/bin/amixer sset Master ' + str(val) + '%', shell=True, stdout=subprocess.PIPE)
	proc.wait()

# print("Current volume: ", get_volume())
# set_volume(0)
# print("Current volume (changed): ", get_volume())
# set_volume(100)
# print("Current volume (changed): ", get_volume())


wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


detector = htm.HandDetector(detectionConfidence=0.7)

vol = 0
volBar = 400

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        #print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        # Center of the line
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Length of the line
        length = math.hypot(x2 - x1, y2 - y1)

        #### Hand range 50 - 300
        #### Volume range 0 - 100
        # Converting hand range to volume range with numpy
        vol = np.interp(length, [50, 300], [0, 100])
        volBar = np.interp(length, [50, 300], [400, 150])
        set_volume(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (40, 450), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
