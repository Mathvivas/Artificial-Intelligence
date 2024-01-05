import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode=False, max_faces=2,
                 refine_landmarks=False,
                 minDetectionConfidence=0.5,
                 minTrackConfidence=0.5):
        self.mode = mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackConfidence = minTrackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.max_faces,
                                                 self.refine_landmarks,
                                                 self.minDetectionConfidence,
                                                 self.minTrackConfidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        img = cv2.resize(img, (800, 600))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, 
                                        self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec, self.drawSpec)
                face = []
                # Every point on the face
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    # Getting values as pixels
                    x, y = int(lm.x * w), int(lm.y * h)
                    # cv2.putText(img, str(id), (x, y),
                    #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img

def main():
    cap = cv2.VideoCapture('videos/1.mp4')
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()