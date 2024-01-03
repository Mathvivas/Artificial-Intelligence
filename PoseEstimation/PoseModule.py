import cv2
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode=False, modelComplexity=1, smooth=True, 
                 enableSegmentation=False, smoothSegmentation=True,
                 detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.smooth, 
                                     self.enableSegmentation, self.smoothSegmentation,
                                     self.detectionConfidence, self.trackConfidence)
        
    def findPose(self, img, draw=True):
        img = cv2.resize(img, (800, 600))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, 
                                        self.mpPose.POSE_CONNECTIONS)
        return img


def main():
    cap = cv2.VideoCapture('videos/4.mp4')
    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()