import cv2,time,os
import numpy as np
import hand_tracking_module as htm


#############################
brush_thickness =30
eraser_thickness =100
#############################

folder_path = "/Users/yaseerarafatkhan/Documents/Computer Vision/8) Virtual Painter/Header Images"
my_list = os.listdir(folder_path)
print(my_list)
overlay_list = []
for im_path in my_list:
    image = cv2.imread(f"{folder_path}/{im_path}")
    overlay_list.append(image)
print(len(overlay_list))        
header = overlay_list[3]
draw_color = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector()

img_canvas = np.zeros((720,1280,3),np.uint8)

xp,yp =0, 0
while True:
    # 1.Import Image
    success,img = cap.read()
    img = cv2.flip(img,1)
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lm_list = detector.findPosition(img,draw=False)
    
    if len(lm_list)!=0:
        
        #print(lm_list)
        
        # tip of Index and middle fingers
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
    
        # 3. Check which Fingers are up
        fingers = detector.fingers_up()
        #print(fingers)
    
        # 4. if selection mode - (Two fingers are up)
        if fingers[1] and fingers[2]:
            xp,yp =0, 0
            #print("selection mode")
            # Checking for the click
            if y1 <125:
                if 250< x1 <450:
                    header = overlay_list[3]
                    draw_color = (255,0,255)
                elif 550< x1 <750:
                    header = overlay_list[1]
                    draw_color = (255,0,0)     
                elif 800 <x1 <950:
                    header = overlay_list[2] 
                    draw_color = (0,255,0)  
                elif 1050 <x1 <1200:
                    header = overlay_list[0] 
                    draw_color =(0,0,0) 
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),draw_color,cv2.FILLED)

        # 5. if Drawing mode - (index finger is up)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,draw_color,cv2.FILLED)
            #print("Drawing mode")
            if xp==0 and yp ==0:
                xp,yp = x1,y1
            if draw_color ==(0,0,0):    
                cv2.line(img,(xp,yp),(x1,y1),draw_color,eraser_thickness)
                cv2.line(img_canvas,(xp,yp),(x1,y1),draw_color,eraser_thickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),draw_color,brush_thickness)
                cv2.line(img_canvas,(xp,yp),(x1,y1),draw_color,brush_thickness)

            xp , yp = x1,y1
    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,img_inv)
    img = cv2.bitwise_or(img,img_canvas)
    # Setting the header Image
    img[0:125,0:1280]= header
    #img = cv2.addWeighted(img,0.5,img_canvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",img_canvas)
    cv2.waitKey(1)