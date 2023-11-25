import cv2
import time
import pose_module as pm


cap =  cv2.VideoCapture("/Users/yaseerarafatkhan/Downloads/Computer Vision/2)Pose Tracking/PoseVideos/2.mp4")
prev_time = 0
detector = pm.pose_detector()
while True:
    success, img = cap.read()
    if not success:
        print("Video has ended.")
        break
    img = detector.find_pose(img)
    lm_list = detector.find_position(img,draw=False)
    if lm_list != 0:
        print(lm_list[14])
        cv2.circle(img,(lm_list[14][1],lm_list[14][2]),20,(255,255,0),cv2.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(img,str(int(fps)),
                (70,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),3)
    
    cv2.imshow("image",img)

    cv2.waitKey(1)