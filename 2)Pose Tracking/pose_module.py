import cv2
import mediapipe as mp
import time

class pose_detector():
    def __init__(self,mode = False, upper_body = False,
                 smooth = True,detection_confidence =False,
                 tracking_confidence =0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode,self.upper_body,
                                      self.smooth,
                                      self.detection_confidence,
                                      self.tracking_confidence)

    def find_pose(self,img,draw = True):
        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results =self.pose.process(img_RGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, 
                                self.results.pose_landmarks,
                                self.mp_pose.POSE_CONNECTIONS)
        return img 
    def find_position(self,img, draw = True):
        lm_list = []
        if self.results.pose_landmarks:       
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        return lm_list
    
    

def main():
    cap =  cv2.VideoCapture("/Users/yaseerarafatkhan/Downloads/Computer Vision/2)Pose Tracking/PoseVideos/2.mp4")
    prev_time = 0
    detector = pose_detector()
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

    
if __name__ =="__main__":
    main()     
    
    