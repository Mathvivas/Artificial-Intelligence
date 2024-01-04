import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/1.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (800, 600))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, 
                                  mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            
            # Every point on the face
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                # Getting values as pixels
                x, y = int(lm.x * w), int(lm.y * h)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)