import cv2
import mediapipe as mp
import time


class Face_Mesh_Detector():
    def __init__(self,static_mode=False,max_faces =2,min_detection_confidence=False,min_tracking_confidence =0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                                self.static_mode,
                                self.max_faces,
                                self.min_detection_confidence,
                                self.min_tracking_confidence)
        self.drawing_spec = self.mp_draw.DrawingSpec(
                                 color=(255, 0, 255), 
                                thickness=1, circle_radius=1)
    
    def find_face_mesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            faces = []
            for faclms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, faclms, 
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            self.drawing_spec,
                            self.drawing_spec)
                face = []
                for id,lm in enumerate(faclms.landmark):
                    #print(lm)
                    ih , iw , ic = img.shape
                    x, y = int(lm.x*iw),int(lm.y*ih)
                    cv2.putText(img, str(id),
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)        
        return img, faces
    

def main():
    
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = Face_Mesh_Detector()
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
    


if __name__ =="__main__":
    main()    