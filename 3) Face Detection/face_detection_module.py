import cv2
import mediapipe as mp
import time

class face_detector():
    def __init__(self,min_detection_con=0.5):
        self.min_detection_con = min_detection_con
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_dectection = self.mp_face_detection.FaceDetection(self.min_detection_con)

    def find_faces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_dectection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                bboxs.append((bbox, detection.score))
                if draw:
                    img = self.fancy_draw(img,bbox)

                    cv2.putText(img, f"Score: {int(detection.score[0] * 100)}%",
                            (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            3, (0, 255, 0), 2)
        return img, bboxs
    
    def fancy_draw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left corner
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top right corner
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom left corner
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom right corner
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

        


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = face_detector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)  # Update img and get bounding boxes
        print(bboxs)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop when 'q' key is pressed

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



if __name__ =="__main__":
    main()    