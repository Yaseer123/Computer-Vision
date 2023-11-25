import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap =  cv2.VideoCapture("/Users/yaseerarafatkhan/Downloads/Computer Vision/2)Pose Tracking/PoseVideos/2.mp4")
prev_time = 0
while True:
    success, img = cap.read()
    if not success:
        print("Video has ended.")
        break
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =pose.process(img_RGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id,lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
    
    
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(img,str(int(fps)),
                (70,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),3)
    
    cv2.imshow("image",img)

    cv2.waitKey(1)