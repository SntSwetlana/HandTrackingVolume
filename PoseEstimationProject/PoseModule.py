import cv2
import mediapipe as mp
import time

print(cv2.__version__)

class poseDetector():
    def __init__(self, mode=False,
                 complexity=1,
                 smooth=True,
                 enableSegm=False,
                 smoothSegm=True,
                 minDetecConf=0.5,
                 minTrackingConfidence=0.5
                 ):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSegm = enableSegm
        self.smoothSegm = smoothSegm
        self.minDetecConf = minDetecConf
        self.minTrackingConfidence = minTrackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # here we can play new features in models
        self.pose = self.mpPose.Pose(
            self.mode,
            self.complexity,
            self.smooth,
            self.enableSegm,
            self.smoothSegm,
            self.minDetecConf,
            self.minTrackingConfidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =self.pose.process(imgRGB)
                #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c  = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('PoseVideo/01.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()