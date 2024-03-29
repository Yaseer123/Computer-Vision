import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
prev_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_dectection = mp_face_detection.FaceDetection(0.75)


while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_dectection.process(imgRGB)
    #print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mp_draw.draw_detection(img,detection)
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw ,ic =img.shape
            bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                int(bboxC.width * iw),int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f"Score: {int(detection.score[0]*100)}%",
                        (bbox[0],bbox[1]-20),
                cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),2)
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img,f"FPS: {int(fps)}",(20,70),
                cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),2)
    cv2.imshow("Image", img)
    
    cv2.waitKey(1)