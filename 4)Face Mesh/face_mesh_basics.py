import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3)
drawing_spec = mp_draw.DrawingSpec(color=(255, 0, 255), 
                        thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faclms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, faclms, 
                    mp_face_mesh.FACEMESH_TESSELATION,
                    drawing_spec,
                    drawing_spec)
            for id,lm in enumerate(faclms.landmark):
                #print(lm)
                ih , iw , ic = img.shape
                x, y = int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}",
                (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
