import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/2.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (800, 600))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # Draws the rectangle by it self
            #mpDraw.draw_detection(img, detection)
            #print(detection.location_data.relative_bounding_box)
            # xmin: 0.692426323890686
            # ymin: 0.22430972754955292
            # width: 0.18528950214385986
            # height: 0.24705280363559723
            
            # Drawing the rectangle by hand
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)