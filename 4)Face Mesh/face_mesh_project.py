import cv2
import mediapipe as mp
import time
import face_mesh_module as fm
cap = cv2.VideoCapture(0)
p_time = 0
detector = fm.Face_Mesh_Detector()
while True:
    success, img = cap.read()
    img, faces = detector.find_face_mesh(img,False)
    if len(faces)!=0:
        print(faces[0])
        
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}",
                (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

