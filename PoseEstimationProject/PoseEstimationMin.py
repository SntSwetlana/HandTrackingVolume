import cv2

import mediapipe as mp
import time

print(cv2.__version__)
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
# here we can play new features in models
pose = mpPose.Pose(
    # static_image_mode=False,
    # model_complexity=1,
    # smooth_landmarks=True,
    # enable_segmentation=False,
    # smooth_segmentation=True,
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5
)

cap = cv2.VideoCapture('PoseVideo/07.mp4')
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c  = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)